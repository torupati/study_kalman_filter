import numpy as np
import pytest

from linear_kf.imu_1d.fgo import (
    FactorGraph,
    build_graph,
    keyframe_indices,
    marginalize,
    position_factor,
    preintegrate,
    prior_factor,
    run_batch_fgo,
    run_sliding_window_fgo,
    solve_window,
)
from linear_kf.imu_1d.kf import KalmanFilterImu1d, rts_smooth, run_filter, transition
from linear_kf.imu_1d.simulator import SimConfig, simulate

X0 = np.array([0.0, 0.0])
P0 = np.diag([1.0, 4.0])


def _run_all(cfg: SimConfig, seed: int = 0):
    sim = simulate(cfg, np.random.default_rng(seed))
    kf = KalmanFilterImu1d(cfg.accel_noise_std, cfg.pos_noise_std)
    res = run_filter(kf, sim.t, sim.acc_meas, sim.pos_meas, X0, P0)
    smooth = rts_smooth(kf, sim.t, sim.acc_meas, res)
    fgo = run_batch_fgo(sim.t, sim.acc_meas, sim.pos_meas, cfg.accel_noise_std, cfg.pos_noise_std, X0, P0)
    return sim, res, smooth, fgo


def _run_fgo(cfg: SimConfig, seed: int = 0):
    sim = simulate(cfg, np.random.default_rng(seed))
    return sim, run_batch_fgo(sim.t, sim.acc_meas, sim.pos_meas, cfg.accel_noise_std, cfg.pos_noise_std, X0, P0)


def test_preintegration_equals_chained_predict():
    cfg = SimConfig(duration=5.0)
    sim = simulate(cfg, np.random.default_rng(0))
    kf = KalmanFilterImu1d(cfg.accel_noise_std, cfg.pos_noise_std)
    i, j = 37, 290
    x, P = np.array([1.5, -0.7]), np.array([[0.3, 0.05], [0.05, 0.2]])
    x_pred, P_pred = x, P
    for k in range(i, j):
        x_pred, P_pred = kf.predict(x_pred, P_pred, sim.acc_meas[k], sim.t[k + 1] - sim.t[k])

    pre = preintegrate(sim.t, sim.acc_meas, i, j, cfg.accel_noise_std)
    np.testing.assert_allclose(pre.F, [[1.0, sim.t[j] - sim.t[i]], [0.0, 1.0]], atol=1e-12)
    np.testing.assert_allclose(pre.F @ x + pre.delta, x_pred, atol=1e-12)
    np.testing.assert_allclose(pre.F @ P @ pre.F.T + pre.cov, P_pred, atol=1e-12)


def test_single_node_graph_is_the_kf_update():
    """A prior plus one position factor is exactly one Kalman measurement update."""
    kf = KalmanFilterImu1d(accel_noise_std=0.1, pos_noise_std=0.5)
    graph = FactorGraph(1)
    graph.add(prior_factor(0, [0.4, 1.0], P0))
    graph.add(position_factor(0, 1.3, 0.5))
    sol = graph.solve()
    x, P, _, _ = kf.update(np.array([0.4, 1.0]), P0, 1.3)
    np.testing.assert_allclose(sol.x[0], x, atol=1e-12)
    np.testing.assert_allclose(sol.marginal(0), P, atol=1e-12)
    assert sol.dof == 1  # 3 residual rows (prior 2 + position 1), 2 unknowns


@pytest.mark.parametrize(
    "cfg",
    [
        SimConfig(),
        SimConfig(scenario="stop_and_go", pos_outage=(30.0, 50.0)),
        SimConfig(duration=20.0, pos_rate=10.0, accel_bias=0.05),
    ],
    ids=["sine", "stop_and_go_outage", "10hz_bias"],
)
def test_batch_fgo_equals_rts_smoother(cfg):
    sim, res, smooth, fgo = _run_all(cfg)
    np.testing.assert_allclose(fgo.x, smooth.x, atol=1e-8)
    np.testing.assert_allclose(fgo.P, smooth.P, atol=1e-10)
    # the newest state has seen the same measurements in both, so FGO's last node is the KF posterior
    np.testing.assert_allclose(fgo.x[-1], res.x[-1], atol=1e-8)
    np.testing.assert_allclose(fgo.P[-1], res.P[-1], atol=1e-10)


def test_smoothing_beats_filtering():
    cfg = SimConfig(scenario="stop_and_go", pos_outage=(30.0, 50.0))
    sim, res, smooth, fgo = _run_all(cfg)
    m = sim.t >= 5.0
    rmse = lambda x: np.sqrt(np.mean((x[m, 0] - sim.pos_true[m]) ** 2))  # noqa: E731
    assert rmse(fgo.x) < rmse(res.x)
    assert np.all(fgo.P[:, 0, 0] <= res.P[:, 0, 0] + 1e-12)
    assert np.all(fgo.P[:, 1, 1] <= res.P[:, 1, 1] + 1e-12)


def test_fgo_chi2_matches_dof():
    """With the right model the whitened cost at the optimum is chi-square with dof degrees of freedom."""
    chi2, dof = [], []
    for seed in range(20):
        _, fgo = _run_fgo(SimConfig(), seed)
        chi2.append(fgo.solution.chi2)
        dof.append(fgo.solution.dof)
    assert dof[0] == 61  # 61 position + 60 IMU (2-dim) + 1 prior (2-dim) residuals = 183 rows, 61 nodes x 2 states
    assert abs(np.mean(chi2) / dof[0] - 1.0) < 3.0 * np.sqrt(2.0 / (dof[0] * len(chi2)))


def test_fgo_error_is_consistent_with_its_covariance():
    errs = []
    for seed in range(10):
        sim, fgo = _run_fgo(SimConfig(), seed)
        e = np.stack([fgo.x[:, 0] - sim.pos_true, fgo.x[:, 1] - sim.vel_true], axis=1)
        errs.append(np.einsum("na,nab,nb->n", e, np.linalg.inv(fgo.P), e))
    assert abs(np.mean(errs) / 2.0 - 1.0) < 0.15  # NEES of a 2-state estimate averages 2


def test_keyframes_are_fixes_plus_ends_and_too_dense_fixes_rejected():
    pos = np.full(11, np.nan)
    pos[[3, 6]] = 1.0
    np.testing.assert_array_equal(keyframe_indices(pos), [0, 3, 6, 10])
    pos[7] = 1.0
    with pytest.raises(ValueError, match="two IMU samples apart"):
        keyframe_indices(pos)


def test_marginalizing_a_node_keeps_the_joint_marginal_of_the_rest():
    cfg = SimConfig(duration=6.0)
    sim = simulate(cfg, np.random.default_rng(3))
    graph, _ = build_graph(sim.t, sim.acc_meas, sim.pos_meas, cfg.accel_noise_std, cfg.pos_noise_std, X0, P0)
    full = graph.solve()
    nodes = list(range(graph.n_nodes))
    x = dict(enumerate(full.x))

    # marginalize an inner node: its two neighbours get one dense prior, which correlates them
    factors = marginalize(graph.factors, 3, x)
    kept = [n for n in nodes if n != 3]
    sol = solve_window(factors, kept, x)
    cols = np.concatenate([np.arange(2 * n, 2 * n + 2) for n in kept])
    np.testing.assert_allclose(sol.x, full.x[kept], atol=1e-10)
    np.testing.assert_allclose(sol.cov, full.cov[np.ix_(cols, cols)], atol=1e-12)


@pytest.mark.parametrize("window", [2, 3])
def test_sliding_window_equals_batch_on_the_data_seen_so_far(window):
    """Linear factors marginalize exactly: each window solve is the batch solution up to its newest keyframe."""
    cfg = SimConfig(duration=12.0, pos_outage=(4.0, 7.0))
    sim = simulate(cfg, np.random.default_rng(2))
    sw = run_sliding_window_fgo(sim.t, sim.acc_meas, sim.pos_meas, cfg.accel_noise_std, cfg.pos_noise_std, X0, P0, window)
    n_key = len(sw.key_idx)
    for g in range(n_key):
        k = sw.key_idx[min(g + window - 1, n_key - 1)]  # newest keyframe of the last solve that held node g
        batch = run_batch_fgo(sim.t[: k + 1], sim.acc_meas[: k + 1], sim.pos_meas[: k + 1], cfg.accel_noise_std, cfg.pos_noise_std, X0, P0)
        np.testing.assert_allclose(sw.x_key[g], batch.solution.x[g], atol=1e-9)
        np.testing.assert_allclose(sw.P_key[g], batch.solution.marginal(g), atol=1e-11)
        if g + window - 1 < n_key:  # the IMU samples after node g came from that same solve
            j = sw.key_idx[g + 1] if g + 1 < n_key else len(sim.t)
            np.testing.assert_allclose(sw.x[sw.key_idx[g] : j], batch.x[sw.key_idx[g] : j], atol=1e-9)


def test_sliding_window_newest_node_is_the_kf_and_full_window_is_batch():
    cfg = SimConfig(scenario="stop_and_go", pos_outage=(30.0, 50.0))
    sim, res, _, fgo = _run_all(cfg)
    args = (sim.t, sim.acc_meas, sim.pos_meas, cfg.accel_noise_std, cfg.pos_noise_std, X0, P0)

    sw = run_sliding_window_fgo(*args, window=4)
    np.testing.assert_allclose(sw.x_online, res.x[sw.key_idx], atol=1e-9)
    np.testing.assert_allclose(sw.P_online, res.P[sw.key_idx], atol=1e-11)

    full = run_sliding_window_fgo(*args, window=len(fgo.key_idx))
    np.testing.assert_allclose(full.x, fgo.x, atol=1e-8)
    np.testing.assert_allclose(full.P, fgo.P, atol=1e-10)


def test_longer_window_lowers_the_fixed_lag_covariance():
    cfg = SimConfig(duration=20.0)
    sim = simulate(cfg, np.random.default_rng(0))
    args = (sim.t, sim.acc_meas, sim.pos_meas, cfg.accel_noise_std, cfg.pos_noise_std, X0, P0)
    sigma = [run_sliding_window_fgo(*args, window=w).P_key[:, 0, 0] for w in (2, 3, 6)]
    assert np.all(sigma[1] <= sigma[0] + 1e-12)
    assert np.all(sigma[2] <= sigma[1] + 1e-12)


def test_sliding_window_needs_two_nodes():
    sim = simulate(SimConfig(duration=3.0), np.random.default_rng(0))
    with pytest.raises(ValueError, match="at least two"):
        run_sliding_window_fgo(sim.t, sim.acc_meas, sim.pos_meas, 1.0, 1.0, X0, P0, window=1)


# --- constant accelerometer bias node ------------------------------------------------------------

BIAS_PRIOR_STD = 0.1


def _augmented_kf_rts(t, acc_meas, pos_meas, accel_noise_std, pos_noise_std, x0, P0, bias_prior_std):
    """Reference: KF + RTS on [p, v, b] with a constant b (no process noise on it), prior b ~ N(0, bias_prior_std^2).

    Returns (filtered x, filtered P, smoothed x, smoothed P), each over all IMU samples.
    """
    n = len(t)
    x = np.r_[x0, 0.0]
    P = np.zeros((3, 3))
    P[:2, :2], P[2, 2] = P0, bias_prior_std**2
    R = pos_noise_std**2
    xf, Pf = np.empty((n, 3)), np.empty((n, 3, 3))
    steps = []
    for k in range(n):
        if not np.isnan(pos_meas[k]):
            K = P[:, 0] / (P[0, 0] + R)
            x = x + K * (pos_meas[k] - x[0])
            I_KH = np.eye(3) - np.outer(K, [1.0, 0.0, 0.0])
            P = I_KH @ P @ I_KH.T + R * np.outer(K, K)
        xf[k], Pf[k] = x, P
        if k < n - 1:
            F, B = transition(t[k + 1] - t[k])
            Fa, Ba = np.eye(3), np.r_[B, 0.0]
            Fa[:2, :2], Fa[:2, 2] = F, -B  # the filter subtracts its bias estimate from each sample
            Qa = accel_noise_std**2 * np.outer(Ba, Ba)
            x, P = Fa @ x + Ba * acc_meas[k], Fa @ P @ Fa.T + Qa
            steps.append((Fa, x, P))
    xs, Ps = xf.copy(), Pf.copy()
    for k in range(n - 2, -1, -1):
        Fa, x_prior, P_prior = steps[k]
        C = np.linalg.solve(P_prior, Fa @ Pf[k]).T
        xs[k] = xf[k] + C @ (xs[k + 1] - x_prior)
        Ps[k] = Pf[k] + C @ (Ps[k + 1] - P_prior) @ C.T
    return xf, Pf, xs, Ps


def _args(sim):
    cfg = sim.config
    return sim.t, sim.acc_meas, sim.pos_meas, cfg.accel_noise_std, cfg.pos_noise_std, X0, P0


def test_preintegration_bias_jacobian_equals_chained_predict_with_bias_removed():
    cfg = SimConfig(duration=5.0)
    sim = simulate(cfg, np.random.default_rng(0))
    kf = KalmanFilterImu1d(cfg.accel_noise_std, cfg.pos_noise_std)
    i, j, b = 37, 290, 0.3
    x = np.array([1.5, -0.7])
    for k in range(i, j):
        x, _ = kf.predict(x, np.eye(2), sim.acc_meas[k] - b, sim.t[k + 1] - sim.t[k])
    pre = preintegrate(sim.t, sim.acc_meas, i, j, cfg.accel_noise_std)
    np.testing.assert_allclose(pre.F @ np.array([1.5, -0.7]) + pre.delta - pre.J_bias * b, x, atol=1e-12)


@pytest.mark.parametrize(
    "cfg",
    [SimConfig(accel_bias=0.05), SimConfig(scenario="stop_and_go", accel_bias=-0.08, pos_outage=(30.0, 50.0))],
    ids=["sine_bias", "stop_and_go_outage_bias"],
)
def test_batch_fgo_with_bias_node_equals_augmented_rts(cfg):
    sim = simulate(cfg, np.random.default_rng(1))
    xf, Pf, xs, Ps = _augmented_kf_rts(*_args(sim), BIAS_PRIOR_STD)
    fgo = run_batch_fgo(*_args(sim), bias_prior_std=BIAS_PRIOR_STD)
    np.testing.assert_allclose(fgo.x, xs[:, :2], atol=1e-8)
    np.testing.assert_allclose(fgo.P, Ps[:, :2, :2], atol=1e-10)
    np.testing.assert_allclose(fgo.bias, xs[0, 2], atol=1e-10)  # the smoothed constant bias
    np.testing.assert_allclose(fgo.bias_std, np.sqrt(Ps[0, 2, 2]), atol=1e-10)
    np.testing.assert_allclose(fgo.x[-1], xf[-1, :2], atol=1e-8)  # newest state: the augmented KF


def test_sliding_window_with_bias_newest_node_is_the_augmented_kf():
    sim = simulate(SimConfig(accel_bias=0.05, pos_outage=(30.0, 50.0)), np.random.default_rng(1))
    xf, Pf, _, _ = _augmented_kf_rts(*_args(sim), BIAS_PRIOR_STD)
    sw = run_sliding_window_fgo(*_args(sim), window=3, bias_prior_std=BIAS_PRIOR_STD)
    np.testing.assert_allclose(sw.x_online, xf[sw.key_idx, :2], atol=1e-9)
    np.testing.assert_allclose(sw.P_online, Pf[sw.key_idx, :2, :2], atol=1e-11)
    np.testing.assert_allclose(sw.bias_online, xf[sw.key_idx, 2], atol=1e-10)
    np.testing.assert_allclose(sw.bias_std_online, np.sqrt(Pf[sw.key_idx, 2, 2]), atol=1e-10)


def test_sliding_window_with_bias_equals_batch_on_the_data_seen_so_far():
    cfg = SimConfig(duration=12.0, accel_bias=0.05, pos_outage=(4.0, 7.0))
    sim = simulate(cfg, np.random.default_rng(2))
    window = 3
    sw = run_sliding_window_fgo(*_args(sim), window=window, bias_prior_std=BIAS_PRIOR_STD)
    n_key = len(sw.key_idx)
    for g in range(n_key):
        k = sw.key_idx[min(g + window - 1, n_key - 1)]  # newest keyframe of the last solve that held node g
        t, acc, pos, *rest = _args(sim)
        batch = run_batch_fgo(t[: k + 1], acc[: k + 1], pos[: k + 1], *rest, bias_prior_std=BIAS_PRIOR_STD)
        np.testing.assert_allclose(sw.x_key[g], batch.solution.values[g], atol=1e-9)
        np.testing.assert_allclose(sw.P_key[g], batch.solution.marginal(g), atol=1e-11)
        if g + window - 1 < n_key:
            j = sw.key_idx[g + 1] if g + 1 < n_key else len(sim.t)
            np.testing.assert_allclose(sw.x[sw.key_idx[g] : j], batch.x[sw.key_idx[g] : j], atol=1e-9)
    final = run_batch_fgo(*_args(sim), bias_prior_std=BIAS_PRIOR_STD)
    np.testing.assert_allclose(sw.bias_online[-1], final.bias, atol=1e-10)


def test_bias_node_recovers_the_bias_and_restores_consistency():
    cfg = SimConfig(accel_bias=0.05)
    z, chi2_bias, chi2_plain = [], [], []
    for seed in range(10):
        sim = simulate(cfg, np.random.default_rng(seed))
        fgo = run_batch_fgo(*_args(sim), bias_prior_std=BIAS_PRIOR_STD)
        z.append((fgo.bias - cfg.accel_bias) / fgo.bias_std)
        chi2_bias.append(fgo.solution.chi2 / fgo.solution.dof)
        plain = run_batch_fgo(*_args(sim))
        chi2_plain.append(plain.solution.chi2 / plain.solution.dof)
    assert np.all(np.abs(z) < 3.0)
    assert abs(np.mean(np.square(z)) - 1.0) < 1.0  # normalized bias error has unit variance (10 samples)
    assert abs(np.mean(chi2_bias) - 1.0) < 0.15
    assert np.mean(chi2_plain) > 1.3  # without the node the unmodelled bias inflates chi2


def test_graph_solution_with_mixed_node_sizes():
    graph = FactorGraph(2, [2, 1])
    graph.add(prior_factor(0, [1.0, 2.0], np.eye(2)))
    graph.add(prior_factor(1, [3.0], [[4.0]]))
    sol = graph.solve()
    np.testing.assert_allclose(sol.values[0], [1.0, 2.0])
    np.testing.assert_allclose(sol.values[1], [3.0])
    np.testing.assert_allclose(sol.marginal(1), [[4.0]])
    np.testing.assert_array_equal(sol.columns(1, 0), [2, 0, 1])
