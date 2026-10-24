"""Batch factor graph optimization (FGO) for the 1D IMU + position problem.

Nodes are keyframe states ``x_i = [p_i, v_i]`` at the position-fix times (plus the first
and last IMU samples), not one per IMU sample. Three kinds of factors connect them:

- ``PriorFactor``     x_0 ~ N(x0, P0), the same prior ``run_filter`` starts from.
- ``PositionFactor``  z_i = p_i + n,  n ~ N(0, R).
- ``ImuFactor``       x_j = F_ij x_i + delta_ij + w,  w ~ N(0, Sigma_ij): the IMU samples between
                      two keyframes, preintegrated. With the zero-order-hold model of ``kf.py``
                      this is exact (it is ``KalmanFilterImu1d.predict`` composed over the interval).

Each factor is whitened by its square-root information ``L`` (``L^T L = cov^-1``) and the stacked
system is solved by Gauss-Newton on the normal equations ``(A^T A) dx = -A^T r``. The problem is
linear, so one iteration gives the exact MAP estimate, and ``(A^T A)^-1`` is its covariance.

Because the model is linear-Gaussian, the batch solution equals the RTS smoother (``kf.rts_smooth``):
the IMU-rate states between keyframes have no measurements, so marginalizing them out of a full-rate
graph gives exactly the preintegrated factors, and ``interpolate`` recovers them by conditioning on
the two neighbouring keyframes.
"""

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from .kf import transition

NODE_DIM = 2


def preintegrate_cumulative(
    t: np.ndarray, acc_meas: np.ndarray, i: int, j: int, accel_noise_std: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preintegrate IMU samples i..j-1 and return (F, delta, cov) from t[i] to every t[m], m = i..j.

    Index 0 of each array is the identity step (m = i), the last is i -> j. The recursion is the
    KF prediction applied to the increment: ``F <- F_k F``, ``delta <- F_k delta + B_k a_k``,
    ``cov <- F_k cov F_k^T + sigma_a^2 B_k B_k^T``.
    """
    m = j - i + 1
    F = np.empty((m, 2, 2))
    delta = np.empty((m, 2))
    cov = np.empty((m, 2, 2))
    F[0], delta[0], cov[0] = np.eye(2), 0.0, 0.0
    for s, k in enumerate(range(i, j), start=1):
        Fk, Bk = transition(t[k + 1] - t[k])
        F[s] = Fk @ F[s - 1]
        delta[s] = Fk @ delta[s - 1] + Bk * acc_meas[k]
        cov[s] = Fk @ cov[s - 1] @ Fk.T + accel_noise_std**2 * np.outer(Bk, Bk)
    return F, delta, cov


@dataclass(frozen=True)
class Preintegration:
    F: np.ndarray  # (2, 2) state transition t_i -> t_j
    delta: np.ndarray  # (2,) IMU-driven increment
    cov: np.ndarray  # (2, 2) accumulated accelerometer noise


def preintegrate(t: np.ndarray, acc_meas: np.ndarray, i: int, j: int, accel_noise_std: float) -> Preintegration:
    F, delta, cov = preintegrate_cumulative(t, acc_meas, i, j, accel_noise_std)
    return Preintegration(F[-1], delta[-1], cov[-1])


def _sqrt_info(cov: np.ndarray) -> np.ndarray:
    """L with L^T L = cov^-1, i.e. L = chol(cov)^-1."""
    try:
        C = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as err:
        raise ValueError(f"factor covariance is not positive definite:\n{cov}") from err
    return np.linalg.solve(C, np.eye(len(cov)))


@dataclass
class Factor:
    """A Gaussian factor on the nodes ``keys``: residual r(x) with covariance ``cov``.

    Subclasses implement ``linearize``, returning the residual at the given node values and its
    Jacobian with respect to each key.
    """

    keys: tuple[int, ...]
    cov: np.ndarray

    @cached_property
    def sqrt_info(self) -> np.ndarray:
        return _sqrt_info(self.cov)

    def linearize(self, xs: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
        raise NotImplementedError


@dataclass
class PriorFactor(Factor):
    mean: np.ndarray = field(kw_only=True)

    def linearize(self, xs):
        return xs[0] - self.mean, [np.eye(NODE_DIM)]


@dataclass
class PositionFactor(Factor):
    z: float = field(kw_only=True)

    def linearize(self, xs):
        return np.array([xs[0][0] - self.z]), [np.array([[1.0, 0.0]])]


@dataclass
class ImuFactor(Factor):
    preint: Preintegration = field(kw_only=True)

    def linearize(self, xs):
        xi, xj = xs
        r = xj - self.preint.F @ xi - self.preint.delta
        return r, [-self.preint.F, np.eye(NODE_DIM)]


def prior_factor(key: int, mean: np.ndarray, cov: np.ndarray) -> PriorFactor:
    return PriorFactor((key,), np.asarray(cov, dtype=float), mean=np.asarray(mean, dtype=float))


def position_factor(key: int, z: float, pos_noise_std: float) -> PositionFactor:
    return PositionFactor((key,), np.array([[pos_noise_std**2]]), z=float(z))


def imu_factor(key_i: int, key_j: int, preint: Preintegration) -> ImuFactor:
    return ImuFactor((key_i, key_j), preint.cov, preint=preint)


@dataclass
class GraphSolution:
    x: np.ndarray  # (n_nodes, 2) MAP estimate
    cov: np.ndarray  # (2 n_nodes, 2 n_nodes) joint covariance (A^T A)^-1
    chi2: float  # sum of squared whitened residuals at x
    dof: int  # residual dimension minus state dimension; E[chi2] = dof when the model is right

    def marginal(self, node: int) -> np.ndarray:
        s = slice(NODE_DIM * node, NODE_DIM * (node + 1))
        return self.cov[s, s]


class FactorGraph:
    def __init__(self, n_nodes: int):
        self.n_nodes = n_nodes
        self.factors: list[Factor] = []

    def add(self, factor: Factor) -> None:
        if any(not 0 <= k < self.n_nodes for k in factor.keys):
            raise ValueError(f"factor keys {factor.keys} out of range for {self.n_nodes} nodes")
        self.factors.append(factor)

    def linearize(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Stack the whitened Jacobian A (rows x 2 n_nodes) and residual r at node values x."""
        rows, rhs = [], []
        for f in self.factors:
            r, Js = f.linearize([x[k] for k in f.keys])
            A = np.zeros((len(r), NODE_DIM * self.n_nodes))
            for k, J in zip(f.keys, Js, strict=True):
                A[:, NODE_DIM * k : NODE_DIM * (k + 1)] = f.sqrt_info @ J
            rows.append(A)
            rhs.append(f.sqrt_info @ r)
        return np.vstack(rows), np.concatenate(rhs)

    def solve(self, x_init: np.ndarray | None = None, iterations: int = 1) -> GraphSolution:
        """Gauss-Newton: x <- x - (A^T A)^-1 A^T r. One iteration is exact for linear factors."""
        x = np.zeros((self.n_nodes, NODE_DIM)) if x_init is None else np.array(x_init, dtype=float)
        for _ in range(iterations):
            A, r = self.linearize(x)
            L = np.linalg.cholesky(A.T @ A)  # information matrix H = L L^T
            dx = -np.linalg.solve(L.T, np.linalg.solve(L, A.T @ r))
            x = x + dx.reshape(self.n_nodes, NODE_DIM)
        A, r = self.linearize(x)
        L_inv = np.linalg.solve(np.linalg.cholesky(A.T @ A), np.eye(A.shape[1]))
        cov = L_inv.T @ L_inv
        return GraphSolution(x, 0.5 * (cov + cov.T), float(r @ r), A.shape[0] - A.shape[1])


def keyframe_indices(pos_meas: np.ndarray) -> np.ndarray:
    """IMU sample indices of the keyframe nodes: every position fix, plus the first and last sample.

    A single IMU step has rank-1 noise (``sigma_a^2 B B^T``), so consecutive keyframes must be at
    least two IMU samples apart for the preintegrated covariance to be invertible.
    """
    n = len(pos_meas)
    idx = np.union1d(np.flatnonzero(~np.isnan(pos_meas)), [0, n - 1])
    if np.any(np.diff(idx) < 2):
        raise ValueError("keyframes must be at least two IMU samples apart (position rate too close to the IMU rate)")
    return idx


def build_graph(
    t: np.ndarray,
    acc_meas: np.ndarray,
    pos_meas: np.ndarray,
    accel_noise_std: float,
    pos_noise_std: float,
    x0: np.ndarray,
    P0: np.ndarray,
) -> tuple[FactorGraph, np.ndarray]:
    """Build the keyframe graph. Returns (graph, keyframe IMU indices); node k is at t[key_idx[k]]."""
    key_idx = keyframe_indices(pos_meas)
    graph = FactorGraph(len(key_idx))
    graph.add(prior_factor(0, x0, P0))
    for node, k in enumerate(key_idx):
        if not np.isnan(pos_meas[k]):
            graph.add(position_factor(node, pos_meas[k], pos_noise_std))
    for node, (i, j) in enumerate(zip(key_idx[:-1], key_idx[1:], strict=True)):
        graph.add(imu_factor(node, node + 1, preintegrate(t, acc_meas, i, j, accel_noise_std)))
    return graph, key_idx


def interpolate(
    t: np.ndarray, acc_meas: np.ndarray, key_idx: np.ndarray, sol: GraphSolution, accel_noise_std: float
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and covariance at every IMU sample, given the keyframe solution.

    Between keyframes i and j the IMU chain is the only information, so x_m is Gaussian given
    (x_i, x_j). With the preintegrated (F_im, delta_im, Sigma_im) from i to m, F_mj = F_ij F_im^-1 and
    Cov(x_m, x_j | x_i) = Sigma_im F_mj^T:

        K = Sigma_im F_mj^T Sigma_ij^-1
        x_m = F_im x_i + delta_im + K (x_j - F_ij x_i - delta_ij) = G [x_i; x_j] + c,  G = [F_im - K F_ij, K]
        P_m = G Cov([x_i; x_j]) G^T + (Sigma_im - K F_mj Sigma_im)
    """
    n = len(t)
    xs = np.empty((n, NODE_DIM))
    Ps = np.empty((n, NODE_DIM, NODE_DIM))
    for node, (i, j) in enumerate(zip(key_idx[:-1], key_idx[1:], strict=True)):
        F_im, d_im, S_im = preintegrate_cumulative(t, acc_meas, i, j, accel_noise_std)
        F_ij, d_ij, S_ij = F_im[-1], d_im[-1], S_im[-1]
        F_mj = F_ij @ np.linalg.inv(F_im)  # (M, 2, 2)
        K = S_im @ F_mj.transpose(0, 2, 1) @ np.linalg.inv(S_ij)
        G = np.concatenate([F_im - K @ F_ij, K], axis=2)  # (M, 2, 4)
        c = d_im - np.einsum("mab,b->ma", K, d_ij)
        s = slice(NODE_DIM * node, NODE_DIM * (node + 2))
        x_pair = sol.x[node : node + 2].reshape(-1)
        P_cond = S_im - K @ F_mj @ S_im
        P = G @ sol.cov[s, s] @ G.transpose(0, 2, 1) + P_cond
        # m = i..j-1; x_j itself is the first sample of the next segment (or written after the loop)
        xs[i:j] = (G @ x_pair + c)[:-1]
        Ps[i:j] = 0.5 * (P + P.transpose(0, 2, 1))[:-1]
    xs[-1], Ps[-1] = sol.x[-1], sol.marginal(len(key_idx) - 1)
    return xs, Ps


@dataclass
class FgoResult:
    x: np.ndarray  # (N, 2) smoothed state at each IMU time
    P: np.ndarray  # (N, 2, 2) its marginal covariance
    key_idx: np.ndarray  # (K,) IMU sample index of each keyframe node
    solution: GraphSolution  # keyframe MAP estimate, joint covariance, chi2


def run_batch_fgo(
    t: np.ndarray,
    acc_meas: np.ndarray,
    pos_meas: np.ndarray,
    accel_noise_std: float,
    pos_noise_std: float,
    x0: np.ndarray,
    P0: np.ndarray,
) -> FgoResult:
    """Solve the whole keyframe graph at once. Same inputs as ``run_filter``; pos_meas is NaN where absent."""
    graph, key_idx = build_graph(t, acc_meas, pos_meas, accel_noise_std, pos_noise_std, x0, P0)
    sol = graph.solve()
    x, P = interpolate(t, acc_meas, key_idx, sol, accel_noise_std)
    return FgoResult(x, P, key_idx, sol)
