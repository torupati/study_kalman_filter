import numpy as np
import pytest

from linear_kf.imu_1d.fgo import FactorGraph, keyframe_indices, position_factor, preintegrate, prior_factor, run_batch_fgo
from linear_kf.imu_1d.kf import KalmanFilterImu1d, rts_smooth, run_filter
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
