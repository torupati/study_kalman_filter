import dataclasses

import numpy as np
import pytest

from imu_gnss_ekf_training import EkfConfig, SimulatorConfig, run_filter, simulate_scenario
from imu_gnss_ekf_training.alignment import RigidTransform2d, align_trajectory, fit_rigid_2d
from imu_gnss_ekf_training.ekf import IDX_VX, IDX_VY, IDX_X, IDX_Y, IDX_YAW, wrap_angle
from imu_gnss_ekf_training.run_conditions import load_run_conditions, save_run_conditions

START_POSE = (30.0, -20.0, np.deg2rad(40.0))


def _config(**overrides):
    fields = {"scenario": "demo2", "dt": 0.01, "total_time": 60.0}
    return SimulatorConfig(**{**fields, **overrides})


def test_start_pose_is_a_rigid_transform_of_the_default_trajectory():
    base = simulate_scenario(_config())
    moved = simulate_scenario(_config(initial_position=START_POSE[:2], initial_yaw=START_POSE[2]))
    transform = RigidTransform2d(yaw=START_POSE[2], translation=np.array(START_POSE[:2]))

    xy, yaw = transform.apply(base["truth_states"][:, [IDX_X, IDX_Y]], base["truth_states"][:, IDX_YAW])
    np.testing.assert_allclose(moved["truth_states"][:, [IDX_X, IDX_Y]], xy, atol=1e-9)
    np.testing.assert_allclose(wrap_angle(moved["truth_states"][:, IDX_YAW] - yaw), 0.0, atol=1e-9)
    velocity, _ = RigidTransform2d(yaw=START_POSE[2], translation=np.zeros(2)).apply(base["truth_states"][:, [IDX_VX, IDX_VY]])
    np.testing.assert_allclose(moved["truth_states"][:, [IDX_VX, IDX_VY]], velocity, atol=1e-9)
    # body-frame sensors do not see the start pose
    np.testing.assert_array_equal(moved["imu_measurements"], base["imu_measurements"])
    np.testing.assert_allclose(moved["vo_delta_measurements"], base["vo_delta_measurements"], atol=1e-9)


def test_fit_rigid_2d_recovers_known_transform():
    rng = np.random.default_rng(0)
    source = rng.normal(size=(50, 2)) * 10.0
    true = RigidTransform2d(yaw=np.deg2rad(-130.0), translation=np.array([4.0, -7.0]))
    target, _ = true.apply(source)
    fitted = fit_rigid_2d(source, target)
    assert fitted.yaw == pytest.approx(true.yaw)
    np.testing.assert_allclose(fitted.translation, true.translation, atol=1e-9)


def test_vo_only_estimate_is_rotated_by_the_unknown_start_pose():
    sim_config = _config(initial_position=START_POSE[:2], initial_yaw=START_POSE[2])
    sim = simulate_scenario(sim_config)
    ekf_config = EkfConfig(
        accel_noise_std=sim_config.accel_noise_std,
        gyro_noise_std=sim_config.gyro_noise_std,
        initial_position_std=0.01,
        initial_yaw_std_rad=np.deg2rad(0.1),
    )
    result = run_filter(
        imu_measurements=sim["imu_measurements"],
        gnss_measurements=sim["gnss_measurements"],
        gnss_available=np.zeros_like(sim["gnss_available"]),
        dt=sim["dt"],
        config=ekf_config,
        vo_from_index=sim["vo_from_index"],
        vo_to_index=sim["vo_to_index"],
        vo_measurements=sim["vo_delta_measurements"],
        vo_covariances=sim["vo_covariances"],
    )
    estimates, truth = result["state_estimates"], sim["truth_states"]
    np.testing.assert_allclose(estimates[0, [IDX_X, IDX_Y, IDX_YAW]], 0.0, atol=1e-9)  # nav frame = start pose

    alignment = align_trajectory(estimates[:, [IDX_X, IDX_Y]], estimates[:, IDX_YAW], truth[:, [IDX_X, IDX_Y]], truth[:, IDX_YAW])
    assert alignment.anchor.yaw == pytest.approx(START_POSE[2])
    assert np.rad2deg(abs(wrap_angle(alignment.fitted.yaw - START_POSE[2]))) < 2.0
    np.testing.assert_allclose(alignment.fitted.translation, START_POSE[:2], atol=1.0)
    assert alignment.ate_anchored_rmse < 1.0
    assert np.linalg.norm(estimates[:, [IDX_X, IDX_Y]] - truth[:, [IDX_X, IDX_Y]], axis=1).mean() > 10.0


def test_run_conditions_round_trip_start_pose(tmp_path):
    config = dataclasses.replace(_config(), initial_position=START_POSE[:2], initial_yaw=START_POSE[2])
    path = save_run_conditions(tmp_path / "run_conditions.toml", config, EkfConfig())
    assert load_run_conditions(path)[0] == config
