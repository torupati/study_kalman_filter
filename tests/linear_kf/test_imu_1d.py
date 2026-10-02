from dataclasses import replace

import numpy as np

from linear_kf.imu_1d.kf import KalmanFilterImu1d, run_filter, run_position_only_filter, transition
from linear_kf.imu_1d.simulator import SimConfig, simulate

X0 = np.array([0.0, 0.0])
P0 = np.diag([1.0, 4.0])


def test_simulator_rates_and_noise_free_measurements():
    cfg = SimConfig(duration=10.0, imu_rate=50.0, pos_rate=2.0, accel_noise_density=0.0, pos_noise_std=0.0)
    sim = simulate(cfg, np.random.default_rng(0))

    assert len(sim.t) == 501
    np.testing.assert_allclose(np.diff(sim.t), 0.02)
    assert sim.pos_available.sum() == 21
    np.testing.assert_array_equal(np.flatnonzero(sim.pos_available), np.arange(0, 501, 25))
    np.testing.assert_allclose(sim.acc_meas, sim.acc_true)
    np.testing.assert_allclose(sim.pos_meas[sim.pos_available], sim.pos_true[sim.pos_available])


def test_simulator_outage_and_bias():
    cfg = SimConfig(duration=20.0, accel_noise_density=0.0, accel_bias=0.1, pos_outage=(5.0, 10.0))
    sim = simulate(cfg, np.random.default_rng(0))

    t_meas = sim.t[sim.pos_available]
    assert not np.any((t_meas >= 5.0) & (t_meas < 10.0))
    assert np.any(np.isclose(t_meas, 10.0))
    np.testing.assert_allclose(sim.acc_meas - sim.acc_true, 0.1)


def test_accel_noise_std_is_density_over_sqrt_dt():
    cfg = SimConfig(imu_rate=100.0, accel_noise_density=0.05)
    np.testing.assert_allclose(cfg.accel_noise_std, 0.05 * np.sqrt(100.0))


def test_Q_is_input_noise_through_B():
    kf = KalmanFilterImu1d(accel_noise_std=0.3, pos_noise_std=0.5)
    dt = 0.1
    _, B = transition(dt)
    np.testing.assert_allclose(kf.Q(dt), 0.09 * np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]]))
    np.testing.assert_allclose(kf.Q(dt), 0.09 * np.outer(B, B))


def test_predict_reproduces_simulator_kinematics_without_noise():
    cfg = SimConfig(duration=10.0, accel_noise_density=0.0, pos_noise_std=0.0)
    sim = simulate(cfg, np.random.default_rng(0))
    kf = KalmanFilterImu1d(accel_noise_std=1e-6, pos_noise_std=1.0)

    x, P = np.array([cfg.initial_position, cfg.initial_velocity]), np.eye(2) * 1e-12
    for k in range(len(sim.t) - 1):
        x, P = kf.predict(x, P, sim.acc_meas[k], cfg.dt)
    np.testing.assert_allclose(x, [sim.pos_true[-1], sim.vel_true[-1]], atol=1e-9)


def test_filter_is_consistent_mean_nis_near_one():
    cfg = SimConfig()
    nis = []
    for seed in range(10):
        sim = simulate(cfg, np.random.default_rng(seed))
        res = run_filter(KalmanFilterImu1d(cfg.accel_noise_std, cfg.pos_noise_std), sim.t, sim.acc_meas, sim.pos_meas, X0, P0)
        nis.append(res.nis[~np.isnan(res.nis)])
    assert 0.8 < np.mean(np.concatenate(nis)) < 1.2


def test_unmodelled_bias_inflates_nis():
    cfg = replace(SimConfig(), accel_bias=0.05)
    sim = simulate(cfg, np.random.default_rng(0))
    res = run_filter(KalmanFilterImu1d(cfg.accel_noise_std, cfg.pos_noise_std), sim.t, sim.acc_meas, sim.pos_meas, X0, P0)
    assert np.nanmean(res.nis) > 1.5


def test_imu_filter_beats_position_only_on_velocity():
    cfg = SimConfig()
    sim = simulate(cfg, np.random.default_rng(0))
    res = run_filter(KalmanFilterImu1d(cfg.accel_noise_std, cfg.pos_noise_std), sim.t, sim.acc_meas, sim.pos_meas, X0, P0)
    res_pos = run_position_only_filter(sim.t, sim.pos_meas, 1.0, cfg.pos_noise_std, X0, P0)
    m = sim.t >= 5.0

    def vel_rmse(r):
        return np.sqrt(np.mean((r.x[m, 1] - sim.vel_true[m]) ** 2))

    assert vel_rmse(res) < 0.5 * vel_rmse(res_pos)
