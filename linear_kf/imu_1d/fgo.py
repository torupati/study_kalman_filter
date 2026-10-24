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

``run_sliding_window_fgo`` solves the same graph online over the most recent ``window`` keyframes.
When the window is full the oldest node is marginalized: the factors touching it are replaced by
one dense ``MarginalFactor`` on its neighbours, from the Schur complement of its information block.
For linear factors this is exact, so every window solve equals the batch solution on the data seen
so far: the newest node is the KF posterior, older ones are fixed-lag smoothed.
"""

from dataclasses import dataclass, field, replace
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


@dataclass
class MarginalFactor(Factor):
    """Dense Gaussian prior on ``keys`` left by marginalizing a node out; already whitened (cov = I).

    r(x) = sum_k J_k (x_k - x_lin_k) + r_lin, so with d = x_keys - x_lin, ||r||^2 = d^T H d + 2 g^T d + const
    for the Schur-complement information H = J^T J and gradient g = J^T r_lin.
    """

    jacobians: list[np.ndarray] = field(kw_only=True)
    x_lin: np.ndarray = field(kw_only=True)  # (len(keys), 2) linearization point
    r_lin: np.ndarray = field(kw_only=True)

    def linearize(self, xs):
        r = self.r_lin + sum(J @ (x - x0) for J, x, x0 in zip(self.jacobians, xs, self.x_lin, strict=True))
        return r, self.jacobians


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


def interpolate_segment(
    t: np.ndarray, acc_meas: np.ndarray, i: int, j: int, x_pair: np.ndarray, cov_pair: np.ndarray, accel_noise_std: float
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and covariance at IMU samples m = i..j, given the joint estimate of the keyframes at i and j.

    Between keyframes i and j the IMU chain is the only information, so x_m is Gaussian given
    (x_i, x_j). With the preintegrated (F_im, delta_im, Sigma_im) from i to m, F_mj = F_ij F_im^-1 and
    Cov(x_m, x_j | x_i) = Sigma_im F_mj^T:

        K = Sigma_im F_mj^T Sigma_ij^-1
        x_m = F_im x_i + delta_im + K (x_j - F_ij x_i - delta_ij) = G [x_i; x_j] + c,  G = [F_im - K F_ij, K]
        P_m = G Cov([x_i; x_j]) G^T + (Sigma_im - K F_mj Sigma_im)

    ``x_pair`` is (2, 2) (rows x_i, x_j) and ``cov_pair`` their (4, 4) joint covariance.
    """
    F_im, d_im, S_im = preintegrate_cumulative(t, acc_meas, i, j, accel_noise_std)
    F_ij, d_ij, S_ij = F_im[-1], d_im[-1], S_im[-1]
    F_mj = F_ij @ np.linalg.inv(F_im)  # (M, 2, 2)
    K = S_im @ F_mj.transpose(0, 2, 1) @ np.linalg.inv(S_ij)
    G = np.concatenate([F_im - K @ F_ij, K], axis=2)  # (M, 2, 4)
    c = d_im - np.einsum("mab,b->ma", K, d_ij)
    P = G @ cov_pair @ G.transpose(0, 2, 1) + (S_im - K @ F_mj @ S_im)
    return G @ np.reshape(x_pair, -1) + c, 0.5 * (P + P.transpose(0, 2, 1))


def interpolate(
    t: np.ndarray, acc_meas: np.ndarray, key_idx: np.ndarray, sol: GraphSolution, accel_noise_std: float
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and covariance at every IMU sample, given the keyframe solution (see ``interpolate_segment``)."""
    n = len(t)
    xs = np.empty((n, NODE_DIM))
    Ps = np.empty((n, NODE_DIM, NODE_DIM))
    for node, (i, j) in enumerate(zip(key_idx[:-1], key_idx[1:], strict=True)):
        s = slice(NODE_DIM * node, NODE_DIM * (node + 2))
        x, P = interpolate_segment(t, acc_meas, i, j, sol.x[node : node + 2], sol.cov[s, s], accel_noise_std)
        # m = i..j-1; x_j itself is the first sample of the next segment (or written after the loop)
        xs[i:j], Ps[i:j] = x[:-1], P[:-1]
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


def marginalize(factors: list[Factor], node: int, x: dict[int, np.ndarray]) -> list[Factor]:
    """Remove ``node``: replace the factors touching it by one ``MarginalFactor`` on their other keys.

    The touching factors, linearized at ``x`` (node id -> value), give the whitened system ``A d = -r``
    over [node, kept...]. With H = A^T A and g = A^T r, the Schur complement

        H' = H_kk - H_km H_mm^-1 H_mk,   g' = g_k - H_km H_mm^-1 g_m

    is the information left on the kept nodes. Writing H' = L L^T, the factor's residual is
    ``L^T d + L^-1 g'``. Factor keys are node ids, not positions in a graph.
    """
    touching = [f for f in factors if node in f.keys]
    rest = [f for f in factors if node not in f.keys]
    keep = sorted({k for f in touching for k in f.keys} - {node})
    if not keep:
        return rest
    order = [node, *keep]
    col = {k: c for c, k in enumerate(order)}
    rows, rhs = [], []
    for f in touching:
        r, Js = f.linearize([x[k] for k in f.keys])
        A = np.zeros((len(r), NODE_DIM * len(order)))
        for k, J in zip(f.keys, Js, strict=True):
            A[:, NODE_DIM * col[k] : NODE_DIM * (col[k] + 1)] = f.sqrt_info @ J
        rows.append(A)
        rhs.append(f.sqrt_info @ r)
    A, r = np.vstack(rows), np.concatenate(rhs)
    H, g = A.T @ A, A.T @ r
    m, k = slice(0, NODE_DIM), slice(NODE_DIM, None)
    H_k = H[k, k] - H[k, m] @ np.linalg.solve(H[m, m], H[m, k])
    g_k = g[k] - H[k, m] @ np.linalg.solve(H[m, m], g[m])
    L = np.linalg.cholesky(0.5 * (H_k + H_k.T))
    J = L.T
    jacobians = [J[:, NODE_DIM * c : NODE_DIM * (c + 1)] for c in range(len(keep))]
    prior = MarginalFactor(
        tuple(keep),
        np.eye(len(J)),
        jacobians=jacobians,
        x_lin=np.array([x[n] for n in keep]),
        r_lin=np.linalg.solve(L, g_k),
    )
    return [*rest, prior]


def solve_window(factors: list[Factor], nodes: list[int], x: dict[int, np.ndarray]) -> GraphSolution:
    """Solve the factors over node ids ``nodes`` (in that order), starting from ``x``."""
    local = {n: i for i, n in enumerate(nodes)}
    graph = FactorGraph(len(nodes))
    for f in factors:
        graph.add(replace(f, keys=tuple(local[k] for k in f.keys)))
    return graph.solve(np.array([x[n] for n in nodes]))


@dataclass
class SlidingWindowResult:
    x: np.ndarray  # (N, 2) at each IMU time, from the last window solve that held both keyframes around it
    P: np.ndarray  # (N, 2, 2)
    key_idx: np.ndarray  # (K,) IMU sample index of each keyframe node
    x_key: np.ndarray  # (K, 2) each keyframe in the last window solve that held it (lag up to window - 1)
    P_key: np.ndarray  # (K, 2, 2)
    x_online: np.ndarray  # (K, 2) each keyframe when it was the newest node (equals the KF posterior)
    P_online: np.ndarray  # (K, 2, 2)


def run_sliding_window_fgo(
    t: np.ndarray,
    acc_meas: np.ndarray,
    pos_meas: np.ndarray,
    accel_noise_std: float,
    pos_noise_std: float,
    x0: np.ndarray,
    P0: np.ndarray,
    window: int,
) -> SlidingWindowResult:
    """Add keyframes one at a time, solve the newest ``window`` of them, then marginalize the oldest.

    Same inputs as ``run_batch_fgo``. ``window >= 2`` is the number of nodes in each solve; once it
    covers every keyframe nothing is marginalized and the result equals ``run_batch_fgo``.
    """
    if window < 2:
        raise ValueError("window must hold at least two keyframes")
    key_idx = keyframe_indices(pos_meas)
    n, n_key = len(t), len(key_idx)
    xs, Ps = np.empty((n, NODE_DIM)), np.empty((n, NODE_DIM, NODE_DIM))
    x_key, P_key = np.empty((n_key, NODE_DIM)), np.empty((n_key, NODE_DIM, NODE_DIM))
    x_online, P_online = np.empty((n_key, NODE_DIM)), np.empty((n_key, NODE_DIM, NODE_DIM))

    def record(sol: GraphSolution, active: list[int], loc: int) -> None:
        """Store node active[loc] and, if active[loc + 1] exists, the IMU samples up to it."""
        node = active[loc]
        x_key[node], P_key[node] = sol.x[loc], sol.marginal(loc)
        if loc + 1 < len(active):
            i, j = key_idx[node], key_idx[node + 1]
            s = slice(NODE_DIM * loc, NODE_DIM * (loc + 2))
            x, P = interpolate_segment(t, acc_meas, i, j, sol.x[loc : loc + 2], sol.cov[s, s], accel_noise_std)
            xs[i:j], Ps[i:j] = x[:-1], P[:-1]

    factors: list[Factor] = []
    active: list[int] = []
    est: dict[int, np.ndarray] = {}
    for g, k in enumerate(key_idx):
        if g == 0:
            factors.append(prior_factor(0, x0, P0))
            est[0] = np.asarray(x0, dtype=float)
        else:
            pre = preintegrate(t, acc_meas, key_idx[g - 1], k, accel_noise_std)
            factors.append(imu_factor(g - 1, g, pre))
            est[g] = pre.F @ est[g - 1] + pre.delta
        if not np.isnan(pos_meas[k]):
            factors.append(position_factor(g, pos_meas[k], pos_noise_std))
        active.append(g)

        sol = solve_window(factors, active, est)
        est.update(zip(active, sol.x, strict=True))
        x_online[g], P_online[g] = sol.x[-1], sol.marginal(len(active) - 1)
        if len(active) == window and g < n_key - 1:
            record(sol, active, 0)
            factors = marginalize(factors, active.pop(0), est)

    for loc in range(len(active)):
        record(sol, active, loc)
    xs[-1], Ps[-1] = x_key[-1], P_key[-1]
    return SlidingWindowResult(xs, Ps, key_idx, x_key, P_key, x_online, P_online)
