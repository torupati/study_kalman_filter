import numpy as np

from imu_gnss_ekf_training import EkfConfig, SimulatorConfig, run_filter, simulate_scenario
from imu_gnss_ekf_training.ekf import (
    IDX_BAX,
    IDX_BAY,
    IDX_BG,
    IDX_VX,
    IDX_VY,
    IDX_X,
    IDX_Y,
    IDX_YAW,
    ImuGnssEkf,
)


def test_simulator_generates_dropout_and_expected_shapes():
    scenario = simulate_scenario(
        SimulatorConfig(total_time=12.0, dt=0.1, gnss_period=0.5, gnss_dropout_probability=0.4, seed=11)
    )

    num_steps = len(scenario["time"])
    assert scenario["truth_states"].shape == (num_steps, 8)
    assert scenario["imu_measurements"].shape == (num_steps, 3)
    assert scenario["gnss_measurements"].shape == (num_steps, 4)
    assert scenario["gnss_available"].shape == (num_steps,)
    assert np.isnan(scenario["gnss_measurements"][~scenario["gnss_available"]]).all()
    assert np.count_nonzero(~scenario["gnss_available"]) > 0


def test_ekf_runs_and_tracks_state_reasonably():
    simulator_config = SimulatorConfig(total_time=40.0, dt=0.1, gnss_period=0.5, gnss_dropout_probability=0.15, seed=7)
    scenario = simulate_scenario(simulator_config)
    result = run_filter(
        imu_measurements=scenario["imu_measurements"],
        gnss_measurements=scenario["gnss_measurements"],
        gnss_available=scenario["gnss_available"],
        dt=scenario["dt"],
        config=EkfConfig(
            accel_noise_std=simulator_config.accel_noise_std,
            gyro_noise_std=simulator_config.gyro_noise_std,
            accel_bias_walk_std=simulator_config.accel_bias_walk_std,
            gyro_bias_walk_std=simulator_config.gyro_bias_walk_std,
            gnss_position_std=simulator_config.gnss_position_std,
            gnss_velocity_std=simulator_config.gnss_velocity_std,
        ),
    )

    estimates = result["state_estimates"]
    truth = scenario["truth_states"]

    assert np.isfinite(estimates).all()
    assert np.isfinite(result["covariances"]).all()

    final_position_error = np.linalg.norm(estimates[-1, [IDX_X, IDX_Y]] - truth[-1, [IDX_X, IDX_Y]])
    final_velocity_error = np.linalg.norm(estimates[-1, [IDX_VX, IDX_VY]] - truth[-1, [IDX_VX, IDX_VY]])
    final_accel_bias_error = np.linalg.norm(estimates[-1, [IDX_BAX, IDX_BAY]] - truth[-1, [IDX_BAX, IDX_BAY]])
    final_gyro_bias_error = abs(estimates[-1, IDX_BG] - truth[-1, IDX_BG])

    assert final_position_error < 5.0
    assert final_velocity_error < 1.0
    assert final_accel_bias_error < 0.35
    assert final_gyro_bias_error < 0.05


def test_bias_jacobian_matches_finite_difference_signs():
    config = EkfConfig()
    ekf = ImuGnssEkf(config)
    state = np.array([1.0, -2.0, 0.4, -0.1, 0.6, 0.08, -0.03, 0.01], dtype=float)
    covariance = np.eye(8)
    imu_sample = np.array([0.35, -0.12, 0.2], dtype=float)
    dt = 0.1
    epsilon = 1.0e-6

    base_prediction, _ = ekf.predict(state, covariance, imu_sample, dt)

    perturb_bax = state.copy()
    perturb_bax[IDX_BAX] += epsilon
    prediction_bax, _ = ekf.predict(perturb_bax, covariance, imu_sample, dt)
    derivative_bax = (prediction_bax - base_prediction) / epsilon

    perturb_bay = state.copy()
    perturb_bay[IDX_BAY] += epsilon
    prediction_bay, _ = ekf.predict(perturb_bay, covariance, imu_sample, dt)
    derivative_bay = (prediction_bay - base_prediction) / epsilon

    yaw = state[IDX_YAW]
    c = np.cos(yaw)
    s = np.sin(yaw)
    expected_bax = np.array([-0.5 * dt**2 * c, -0.5 * dt**2 * s, -dt * c, -dt * s])
    expected_bay = np.array([0.5 * dt**2 * s, -0.5 * dt**2 * c, dt * s, -dt * c])

    np.testing.assert_allclose(derivative_bax[[IDX_X, IDX_Y, IDX_VX, IDX_VY]], expected_bax, atol=1e-6)
    np.testing.assert_allclose(derivative_bay[[IDX_X, IDX_Y, IDX_VX, IDX_VY]], expected_bay, atol=1e-6)
