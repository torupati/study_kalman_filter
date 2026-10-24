"""Batch factor graph optimization (FGO) for the 1D IMU + position problem.

Nodes are keyframe states ``x_i = [p_i, v_i]`` at the position-fix times (plus the first
and last IMU samples), not one per IMU sample. Three kinds of factors connect them:

- ``PriorFactor``     x_0 ~ N(x0, P0), the same prior ``run_filter`` starts from.
- ``PositionFactor``  z_i = p_i + n,  n ~ N(0, R).
- ``ImuFactor``       x_j = F_ij x_i + delta_ij + w,  w ~ N(0, Sigma_ij): the IMU samples between
                      two keyframes, preintegrated. With the zero-order-hold model of ``kf.py``
                      this is exact (it is ``KalmanFilterImu1d.predict`` composed over the interval).

Optionally (``bias_prior_std``), one more node holds a constant accelerometer bias ``b`` with prior
N(0, bias_prior_std^2). The accelerometer then reads ``a + b + n``, so every IMU factor becomes
``x_j = F_ij x_i + delta_ij - J_ij b + w``, where ``J_ij = d delta_ij / d a^m`` is accumulated during
preintegration, and connects to the bias node as well as its two keyframes.

Each factor is whitened by its square-root information ``L`` (``L^T L = cov^-1``) and the stacked
system is solved by Gauss-Newton on the normal equations ``(A^T A) dx = -A^T r``. The problem is
linear, so one iteration gives the exact MAP estimate, and ``(A^T A)^-1`` is its covariance.

Because the model is linear-Gaussian, the batch solution equals the RTS smoother (``kf.rts_smooth``;
with the bias node, an RTS smoother on the augmented state [p, v, b] with a constant b): the IMU-rate
states between keyframes have no measurements, so marginalizing them out of a full-rate graph gives
exactly the preintegrated factors, and ``interpolate`` recovers them by conditioning on the two
neighbouring keyframes (and the bias).

``run_sliding_window_fgo`` solves the same graph online over the most recent ``window`` keyframes
(the bias node, if any, is always in the window). When the window is full the oldest keyframe is
marginalized: the factors touching it are replaced by one dense ``MarginalFactor`` on its neighbours,
from the Schur complement of its information block. For linear factors this is exact, so every
window solve equals the batch solution on the data seen so far: the newest node is the KF posterior,
older ones are fixed-lag smoothed.
"""

from dataclasses import dataclass, field, replace
from functools import cached_property

import numpy as np

from .kf import transition

NODE_DIM = 2  # keyframe state [p, v]
BIAS_DIM = 1  # constant accelerometer bias b


def preintegrate_cumulative(
    t: np.ndarray, acc_meas: np.ndarray, i: int, j: int, accel_noise_std: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preintegrate IMU samples i..j-1 and return (F, delta, cov, J_bias) from t[i] to every t[m], m = i..j.

    Index 0 of each array is the identity step (m = i), the last is i -> j. The recursion is the
    KF prediction applied to the increment: ``F <- F_k F``, ``delta <- F_k delta + B_k a_k``,
    ``cov <- F_k cov F_k^T + sigma_a^2 B_k B_k^T``. ``J_bias <- F_k J_bias + B_k`` is the increment's
    sensitivity to a constant offset in every accelerometer sample: a bias b changes delta by ``-J_bias b``.
    """
    m = j - i + 1
    F = np.empty((m, 2, 2))
    delta = np.empty((m, 2))
    cov = np.empty((m, 2, 2))
    J = np.empty((m, 2))
    F[0], delta[0], cov[0], J[0] = np.eye(2), 0.0, 0.0, 0.0
    for s, k in enumerate(range(i, j), start=1):
        Fk, Bk = transition(t[k + 1] - t[k])
        F[s] = Fk @ F[s - 1]
        delta[s] = Fk @ delta[s - 1] + Bk * acc_meas[k]
        cov[s] = Fk @ cov[s - 1] @ Fk.T + accel_noise_std**2 * np.outer(Bk, Bk)
        J[s] = Fk @ J[s - 1] + Bk
    return F, delta, cov, J


@dataclass(frozen=True)
class Preintegration:
    F: np.ndarray  # (2, 2) state transition t_i -> t_j
    delta: np.ndarray  # (2,) IMU-driven increment
    cov: np.ndarray  # (2, 2) accumulated accelerometer noise
    J_bias: np.ndarray  # (2,) d delta / d a^m: a constant bias b shifts delta by -J_bias b


def preintegrate(t: np.ndarray, acc_meas: np.ndarray, i: int, j: int, accel_noise_std: float) -> Preintegration:
    F, delta, cov, J = preintegrate_cumulative(t, acc_meas, i, j, accel_noise_std)
    return Preintegration(F[-1], delta[-1], cov[-1], J[-1])


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
        return xs[0] - self.mean, [np.eye(len(self.mean))]


@dataclass
class PositionFactor(Factor):
    z: float = field(kw_only=True)

    def linearize(self, xs):
        return np.array([xs[0][0] - self.z]), [np.array([[1.0, 0.0]])]


@dataclass
class ImuFactor(Factor):
    """On keys (i, j), or (i, j, bias) when the accelerometer bias is estimated."""

    preint: Preintegration = field(kw_only=True)

    def linearize(self, xs):
        xi, xj = xs[0], xs[1]
        r = xj - self.preint.F @ xi - self.preint.delta
        jacobians = [-self.preint.F, np.eye(NODE_DIM)]
        if len(xs) == 3:
            r = r + self.preint.J_bias * xs[2][0]
            jacobians.append(self.preint.J_bias[:, None])
        return r, jacobians


@dataclass
class MarginalFactor(Factor):
    """Dense Gaussian prior on ``keys`` left by marginalizing a node out; already whitened (cov = I).

    r(x) = sum_k J_k (x_k - x_lin_k) + r_lin, so with d = x_keys - x_lin, ||r||^2 = d^T H d + 2 g^T d + const
    for the Schur-complement information H = J^T J and gradient g = J^T r_lin.
    """

    jacobians: list[np.ndarray] = field(kw_only=True)
    x_lin: list[np.ndarray] = field(kw_only=True)  # linearization point, one value per key
    r_lin: np.ndarray = field(kw_only=True)

    def linearize(self, xs):
        r = self.r_lin + sum(J @ (x - x0) for J, x, x0 in zip(self.jacobians, xs, self.x_lin, strict=True))
        return r, self.jacobians


def prior_factor(key: int, mean: np.ndarray, cov: np.ndarray) -> PriorFactor:
    return PriorFactor((key,), np.atleast_2d(np.asarray(cov, dtype=float)), mean=np.atleast_1d(np.asarray(mean, dtype=float)))


def position_factor(key: int, z: float, pos_noise_std: float) -> PositionFactor:
    return PositionFactor((key,), np.array([[pos_noise_std**2]]), z=float(z))


def imu_factor(key_i: int, key_j: int, preint: Preintegration, bias_key: int | None = None) -> ImuFactor:
    keys = (key_i, key_j) if bias_key is None else (key_i, key_j, bias_key)
    return ImuFactor(keys, preint.cov, preint=preint)


def bias_prior_factor(key: int, bias_prior_std: float) -> PriorFactor:
    return prior_factor(key, [0.0], [[bias_prior_std**2]])


@dataclass
class GraphSolution:
    values: list[np.ndarray]  # MAP estimate of each node (length 2 for a keyframe, 1 for the bias)
    cov: np.ndarray  # joint covariance (A^T A)^-1 over all node columns
    chi2: float  # sum of squared whitened residuals at the estimate
    dof: int  # residual dimension minus state dimension; E[chi2] = dof when the model is right
    offsets: np.ndarray  # (n_nodes + 1,) node k occupies columns offsets[k]:offsets[k + 1]

    @property
    def x(self) -> np.ndarray:
        """(n_nodes, 2) estimate, for graphs whose nodes are all keyframes."""
        return np.array(self.values)

    @property
    def flat(self) -> np.ndarray:
        return np.concatenate(self.values)

    def columns(self, *nodes: int) -> np.ndarray:
        return np.concatenate([np.arange(self.offsets[n], self.offsets[n + 1]) for n in nodes])

    def marginal(self, *nodes: int) -> np.ndarray:
        """Joint covariance of the given nodes (one node: its marginal covariance)."""
        c = self.columns(*nodes)
        return self.cov[np.ix_(c, c)]


class FactorGraph:
    def __init__(self, n_nodes: int, dims: list[int] | None = None):
        self.n_nodes = n_nodes
        self.dims = [NODE_DIM] * n_nodes if dims is None else list(dims)
        if len(self.dims) != n_nodes:
            raise ValueError(f"{len(self.dims)} node dimensions for {n_nodes} nodes")
        self.offsets = np.concatenate([[0], np.cumsum(self.dims)]).astype(int)
        self.factors: list[Factor] = []

    def add(self, factor: Factor) -> None:
        if any(not 0 <= k < self.n_nodes for k in factor.keys):
            raise ValueError(f"factor keys {factor.keys} out of range for {self.n_nodes} nodes")
        self.factors.append(factor)

    def linearize(self, x: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Stack the whitened Jacobian A (rows x state columns) and residual r at node values x."""
        rows, rhs = [], []
        for f in self.factors:
            r, Js = f.linearize([x[k] for k in f.keys])
            A = np.zeros((len(r), self.offsets[-1]))
            for k, J in zip(f.keys, Js, strict=True):
                A[:, self.offsets[k] : self.offsets[k + 1]] = f.sqrt_info @ J
            rows.append(A)
            rhs.append(f.sqrt_info @ r)
        return np.vstack(rows), np.concatenate(rhs)

    def solve(self, x_init=None, iterations: int = 1) -> GraphSolution:
        """Gauss-Newton: x <- x - (A^T A)^-1 A^T r. One iteration is exact for linear factors.

        ``x_init`` is one value per node (an (n_nodes, 2) array works for keyframe-only graphs); default zeros.
        """
        if x_init is None:
            x = [np.zeros(d) for d in self.dims]
        else:
            x = [np.array(v, dtype=float).reshape(d) for v, d in zip(x_init, self.dims, strict=True)]
        for _ in range(iterations):
            A, r = self.linearize(x)
            L = np.linalg.cholesky(A.T @ A)  # information matrix H = L L^T
            dx = -np.linalg.solve(L.T, np.linalg.solve(L, A.T @ r))
            x = [v + dx[o : o + len(v)] for v, o in zip(x, self.offsets, strict=False)]
        A, r = self.linearize(x)
        L_inv = np.linalg.solve(np.linalg.cholesky(A.T @ A), np.eye(A.shape[1]))
        cov = L_inv.T @ L_inv
        return GraphSolution(x, 0.5 * (cov + cov.T), float(r @ r), A.shape[0] - A.shape[1], self.offsets.copy())


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
    bias_prior_std: float | None = None,
) -> tuple[FactorGraph, np.ndarray]:
    """Build the keyframe graph. Returns (graph, keyframe IMU indices); node k is at t[key_idx[k]].

    With ``bias_prior_std``, node ``len(key_idx)`` is the accelerometer bias, with prior N(0, bias_prior_std^2).
    """
    key_idx = keyframe_indices(pos_meas)
    n_key = len(key_idx)
    bias_node = None if bias_prior_std is None else n_key
    graph = FactorGraph(n_key, None) if bias_node is None else FactorGraph(n_key + 1, [NODE_DIM] * n_key + [BIAS_DIM])
    graph.add(prior_factor(0, x0, P0))
    if bias_node is not None:
        graph.add(bias_prior_factor(bias_node, bias_prior_std))
    for node, k in enumerate(key_idx):
        if not np.isnan(pos_meas[k]):
            graph.add(position_factor(node, pos_meas[k], pos_noise_std))
    for node, (i, j) in enumerate(zip(key_idx[:-1], key_idx[1:], strict=True)):
        graph.add(imu_factor(node, node + 1, preintegrate(t, acc_meas, i, j, accel_noise_std), bias_node))
    return graph, key_idx


def interpolate_segment(
    t: np.ndarray, acc_meas: np.ndarray, i: int, j: int, y: np.ndarray, cov_y: np.ndarray, accel_noise_std: float
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and covariance at IMU samples m = i..j, given the joint estimate of the keyframes at i and j.

    ``y`` is [x_i; x_j] (4,) or, with a bias node, [x_i; x_j; b] (5,), and ``cov_y`` its joint covariance.
    Between keyframes i and j the IMU chain is the only information, so x_m is Gaussian given
    (x_i, x_j, b). With the preintegrated (F_im, delta_im, Sigma_im, J_im) from i to m,
    F_mj = F_ij F_im^-1 and Cov(x_m, x_j | x_i, b) = Sigma_im F_mj^T:

        K = Sigma_im F_mj^T Sigma_ij^-1
        x_m = F_im x_i + delta_im - J_im b + K (x_j - F_ij x_i - delta_ij + J_ij b) = G y + c,
        G = [F_im - K F_ij, K, -J_im + K J_ij]   (last column only with a bias)
        P_m = G cov_y G^T + (Sigma_im - K F_mj Sigma_im)
    """
    F_im, d_im, S_im, J_im = preintegrate_cumulative(t, acc_meas, i, j, accel_noise_std)
    F_ij, d_ij, S_ij, J_ij = F_im[-1], d_im[-1], S_im[-1], J_im[-1]
    F_mj = F_ij @ np.linalg.inv(F_im)  # (M, 2, 2)
    K = S_im @ F_mj.transpose(0, 2, 1) @ np.linalg.inv(S_ij)
    G = np.concatenate([F_im - K @ F_ij, K], axis=2)  # (M, 2, 4)
    if len(y) == 2 * NODE_DIM + BIAS_DIM:
        G = np.concatenate([G, (K @ J_ij - J_im)[:, :, None]], axis=2)  # (M, 2, 5)
    c = d_im - np.einsum("mab,b->ma", K, d_ij)
    P = G @ cov_y @ G.transpose(0, 2, 1) + (S_im - K @ F_mj @ S_im)
    return G @ np.asarray(y) + c, 0.5 * (P + P.transpose(0, 2, 1))


def interpolate(
    t: np.ndarray, acc_meas: np.ndarray, key_idx: np.ndarray, sol: GraphSolution, accel_noise_std: float, bias_node: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and covariance at every IMU sample, given the keyframe solution (see ``interpolate_segment``)."""
    n = len(t)
    xs = np.empty((n, NODE_DIM))
    Ps = np.empty((n, NODE_DIM, NODE_DIM))
    extra = () if bias_node is None else (bias_node,)
    flat = sol.flat
    for node, (i, j) in enumerate(zip(key_idx[:-1], key_idx[1:], strict=True)):
        cols = sol.columns(node, node + 1, *extra)
        x, P = interpolate_segment(t, acc_meas, i, j, flat[cols], sol.cov[np.ix_(cols, cols)], accel_noise_std)
        # m = i..j-1; x_j itself is the first sample of the next segment (or written after the loop)
        xs[i:j], Ps[i:j] = x[:-1], P[:-1]
    last = len(key_idx) - 1
    xs[-1], Ps[-1] = sol.values[last], sol.marginal(last)
    return xs, Ps


@dataclass
class FgoResult:
    x: np.ndarray  # (N, 2) smoothed state at each IMU time
    P: np.ndarray  # (N, 2, 2) its marginal covariance
    key_idx: np.ndarray  # (K,) IMU sample index of each keyframe node
    solution: GraphSolution  # keyframe (and bias) MAP estimate, joint covariance, chi2
    bias: float | None = None  # accelerometer bias estimate, when it is a node
    bias_std: float | None = None


def run_batch_fgo(
    t: np.ndarray,
    acc_meas: np.ndarray,
    pos_meas: np.ndarray,
    accel_noise_std: float,
    pos_noise_std: float,
    x0: np.ndarray,
    P0: np.ndarray,
    bias_prior_std: float | None = None,
) -> FgoResult:
    """Solve the whole keyframe graph at once. Same inputs as ``run_filter``; pos_meas is NaN where absent.

    ``bias_prior_std`` adds a constant accelerometer bias node with prior N(0, bias_prior_std^2).
    """
    graph, key_idx = build_graph(t, acc_meas, pos_meas, accel_noise_std, pos_noise_std, x0, P0, bias_prior_std)
    sol = graph.solve()
    bias_node = None if bias_prior_std is None else len(key_idx)
    x, P = interpolate(t, acc_meas, key_idx, sol, accel_noise_std, bias_node)
    if bias_node is None:
        return FgoResult(x, P, key_idx, sol)
    return FgoResult(x, P, key_idx, sol, float(sol.values[bias_node][0]), float(np.sqrt(sol.marginal(bias_node)[0, 0])))


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
    offset = dict(zip(order, np.concatenate([[0], np.cumsum([len(x[k]) for k in order])]), strict=False))
    width = offset[order[-1]] + len(x[order[-1]])
    rows, rhs = [], []
    for f in touching:
        r, Js = f.linearize([x[k] for k in f.keys])
        A = np.zeros((len(r), width))
        for k, J in zip(f.keys, Js, strict=True):
            A[:, offset[k] : offset[k] + len(x[k])] = f.sqrt_info @ J
        rows.append(A)
        rhs.append(f.sqrt_info @ r)
    A, r = np.vstack(rows), np.concatenate(rhs)
    H, g = A.T @ A, A.T @ r
    dm = len(x[node])
    m, k = slice(0, dm), slice(dm, None)
    H_k = H[k, k] - H[k, m] @ np.linalg.solve(H[m, m], H[m, k])
    g_k = g[k] - H[k, m] @ np.linalg.solve(H[m, m], g[m])
    L = np.linalg.cholesky(0.5 * (H_k + H_k.T))
    J = L.T
    jacobians = [J[:, offset[n] - dm : offset[n] - dm + len(x[n])] for n in keep]
    prior = MarginalFactor(
        tuple(keep),
        np.eye(len(J)),
        jacobians=jacobians,
        x_lin=[np.array(x[n]) for n in keep],
        r_lin=np.linalg.solve(L, g_k),
    )
    return [*rest, prior]


def solve_window(factors: list[Factor], nodes: list[int], x: dict[int, np.ndarray]) -> GraphSolution:
    """Solve the factors over node ids ``nodes`` (in that order), starting from ``x``."""
    local = {n: i for i, n in enumerate(nodes)}
    graph = FactorGraph(len(nodes), [len(x[n]) for n in nodes])
    for f in factors:
        graph.add(replace(f, keys=tuple(local[k] for k in f.keys)))
    return graph.solve([x[n] for n in nodes])


@dataclass
class SlidingWindowResult:
    x: np.ndarray  # (N, 2) at each IMU time, from the last window solve that held both keyframes around it
    P: np.ndarray  # (N, 2, 2)
    key_idx: np.ndarray  # (K,) IMU sample index of each keyframe node
    x_key: np.ndarray  # (K, 2) each keyframe in the last window solve that held it (lag up to window - 1)
    P_key: np.ndarray  # (K, 2, 2)
    x_online: np.ndarray  # (K, 2) each keyframe when it was the newest node (equals the KF posterior)
    P_online: np.ndarray  # (K, 2, 2)
    bias_online: np.ndarray  # (K,) bias estimate after each keyframe's solve; NaN without a bias node
    bias_std_online: np.ndarray  # (K,)


def run_sliding_window_fgo(
    t: np.ndarray,
    acc_meas: np.ndarray,
    pos_meas: np.ndarray,
    accel_noise_std: float,
    pos_noise_std: float,
    x0: np.ndarray,
    P0: np.ndarray,
    window: int,
    bias_prior_std: float | None = None,
) -> SlidingWindowResult:
    """Add keyframes one at a time, solve the newest ``window`` of them, then marginalize the oldest.

    Same inputs as ``run_batch_fgo``. ``window >= 2`` is the number of keyframes in each solve; once it
    covers every keyframe nothing is marginalized and the result equals ``run_batch_fgo``. The bias
    node (with ``bias_prior_std``) is never marginalized: every IMU factor touches it, so each
    marginalization folds the old keyframe's information about b into the new prior.
    """
    if window < 2:
        raise ValueError("window must hold at least two keyframes")
    key_idx = keyframe_indices(pos_meas)
    n, n_key = len(t), len(key_idx)
    bias_node = None if bias_prior_std is None else n_key
    extra = [] if bias_node is None else [bias_node]
    xs, Ps = np.empty((n, NODE_DIM)), np.empty((n, NODE_DIM, NODE_DIM))
    x_key, P_key = np.empty((n_key, NODE_DIM)), np.empty((n_key, NODE_DIM, NODE_DIM))
    x_online, P_online = np.empty((n_key, NODE_DIM)), np.empty((n_key, NODE_DIM, NODE_DIM))
    bias_online, bias_std_online = np.full(n_key, np.nan), np.full(n_key, np.nan)

    def record(sol: GraphSolution, active: list[int], loc: int) -> None:
        """Store node active[loc] and, if active[loc + 1] exists, the IMU samples up to it."""
        node = active[loc]
        x_key[node], P_key[node] = sol.values[loc], sol.marginal(loc)
        if loc + 1 < len(active):
            i, j = key_idx[node], key_idx[node + 1]
            cols = sol.columns(loc, loc + 1, *([len(active)] if extra else []))  # the bias is solved last
            x, P = interpolate_segment(t, acc_meas, i, j, sol.flat[cols], sol.cov[np.ix_(cols, cols)], accel_noise_std)
            xs[i:j], Ps[i:j] = x[:-1], P[:-1]

    factors: list[Factor] = []
    active: list[int] = []
    est: dict[int, np.ndarray] = {}
    if bias_node is not None:
        factors.append(bias_prior_factor(bias_node, bias_prior_std))
        est[bias_node] = np.zeros(BIAS_DIM)
    for g, k in enumerate(key_idx):
        if g == 0:
            factors.append(prior_factor(0, x0, P0))
            est[0] = np.asarray(x0, dtype=float)
        else:
            pre = preintegrate(t, acc_meas, key_idx[g - 1], k, accel_noise_std)
            factors.append(imu_factor(g - 1, g, pre, bias_node))
            b = 0.0 if bias_node is None else est[bias_node][0]
            est[g] = pre.F @ est[g - 1] + pre.delta - pre.J_bias * b
        if not np.isnan(pos_meas[k]):
            factors.append(position_factor(g, pos_meas[k], pos_noise_std))
        active.append(g)

        sol = solve_window(factors, active + extra, est)
        est.update(zip(active + extra, sol.values, strict=True))
        x_online[g], P_online[g] = sol.values[len(active) - 1], sol.marginal(len(active) - 1)
        if bias_node is not None:
            bias_online[g] = sol.values[-1][0]
            bias_std_online[g] = np.sqrt(sol.marginal(len(active))[0, 0])
        if len(active) == window and g < n_key - 1:
            record(sol, active, 0)
            factors = marginalize(factors, active.pop(0), est)

    for loc in range(len(active)):
        record(sol, active, loc)
    xs[-1], Ps[-1] = x_key[-1], P_key[-1]
    return SlidingWindowResult(xs, Ps, key_idx, x_key, P_key, x_online, P_online, bias_online, bias_std_online)
