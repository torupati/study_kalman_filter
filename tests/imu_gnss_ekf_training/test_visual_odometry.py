import numpy as np
import pytest

from imu_gnss_ekf_training import EkfConfig, SimulatorConfig, run_filter, simulate_scenario
from imu_gnss_ekf_training.ekf import (
    AUGMENTED_STATE_SIZE,
    IDX_CLONE_X,
    IDX_CLONE_Y,
    IDX_CLONE_YAW,
    IDX_X,
    IDX_Y,
    IDX_YAW,
    STATE_SIZE,
    ImuGnssEkf,
    wrap_angle,
)
from imu_gnss_ekf_training.simulator import compose_pose_2d, relative_pose_2d


def test_relative_and_compose_are_inverse_across_yaw_wrap():
    pose_from = np.array([1.0, -2.0, np.deg2rad(170.0)])
    pose_to = np.array([0.5, -1.2, np.deg2rad(-175.0)])
    delta = relative_pose_2d(pose_from, pose_to)
    assert delta[2] == pytest.approx(np.deg2rad(15.0))
    np.testing.assert_allclose(compose_pose_2d(pose_from, delta), pose_to, atol=1e-12)


def test_relative_pose_is_in_previous_body_frame():
    # facing +y, moving 1 m along +y is 1 m straight ahead in the body frame
    delta = relative_pose_2d(np.array([0.0, 0.0, np.pi / 2]), np.array([0.0, 1.0, np.pi / 2]))
    np.testing.assert_allclose(delta, [1.0, 0.0, 0.0], atol=1e-12)


def test_vo_on_circle_matches_speed_and_turn_rate():
    config = SimulatorConfig(scenario="circle", dt=0.01, total_time=30.0, vo_period=0.1)
    sim = simulate_scenario(config)
    assert np.all(sim["vo_to_index"] - sim["vo_from_index"] == 10)
    cruising = sim["time"][sim["vo_from_index"]] > 5.0  # 2 m/s on a 10 m radius after the ramp
    delta = sim["vo_delta_truth"][cruising]
    np.testing.assert_allclose(np.hypot(delta[:, 0], delta[:, 1]), 0.2, rtol=1e-3)
    np.testing.assert_allclose(delta[:, 2], 0.02, rtol=1e-6)


@pytest.mark.parametrize("dropout", [0.0, 0.3])
def test_composing_true_deltas_rebuilds_trajectory(dropout):
    config = SimulatorConfig(scenario="demo2", dt=0.01, total_time=40.0, vo_dropout_probability=dropout)
    sim = simulate_scenario(config)
    frm, to = sim["vo_from_index"], sim["vo_to_index"]
    np.testing.assert_array_equal(frm[1:], to[:-1])  # deltas chain frame to frame
    poses = sim["truth_states"][:, [IDX_X, IDX_Y, IDX_YAW]]
    pose = poses[frm[0]]
    for delta in sim["vo_delta_truth"]:
        pose = compose_pose_2d(pose, delta)
    np.testing.assert_allclose(pose, poses[to[-1]], atol=1e-9)
    if dropout > 0.0:
        assert np.any(to - frm > 10)


def test_vo_noise_is_consistent_with_covariance():
    config = SimulatorConfig(scenario="demo1", dt=0.01, total_time=300.0, seed=1)
    sim = simulate_scenario(config)
    error = sim["vo_delta_measurements"] - sim["vo_delta_truth"]
    error[:, 2] = wrap_angle(error[:, 2])
    normalized = error / np.sqrt(np.diagonal(sim["vo_covariances"], axis1=1, axis2=2))
    np.testing.assert_allclose(normalized.std(axis=0), 1.0, atol=0.05)


def test_vo_does_not_change_imu_gnss_draws():
    base = simulate_scenario(SimulatorConfig(seed=5))
    other = simulate_scenario(SimulatorConfig(seed=5, vo_period=0.3, vo_dropout_probability=0.5, vo_translation_std=0.1))
    for key in ("truth_states", "imu_measurements", "gnss_measurements", "gnss_available"):
        np.testing.assert_array_equal(base[key], other[key])


# --- EKF fusion (stochastic cloning) ---


def _run(sim, sim_config, use_vo, **vo_overrides):
    vo = {}
    if use_vo:
        vo = {
            "vo_from_index": sim["vo_from_index"],
            "vo_to_index": sim["vo_to_index"],
            "vo_measurements": sim["vo_delta_measurements"],
            "vo_covariances": sim["vo_covariances"],
            **vo_overrides,
        }
    return run_filter(
        imu_measurements=sim["imu_measurements"],
        gnss_measurements=sim["gnss_measurements"],
        gnss_available=sim["gnss_available"],
        dt=sim["dt"],
        config=EkfConfig(accel_noise_std=sim_config.accel_noise_std, gyro_noise_std=sim_config.gyro_noise_std),
        **vo,
    )


def test_vo_jacobian_matches_finite_differences():
    rng = np.random.default_rng(0)
    state = rng.normal(size=AUGMENTED_STATE_SIZE)
    _, H = ImuGnssEkf.vo_measurement_model(state)
    eps = 1e-6
    numeric = np.column_stack(
        [
            (ImuGnssEkf.vo_measurement_model(state + eps * e)[0] - ImuGnssEkf.vo_measurement_model(state - eps * e)[0]) / (2 * eps)
            for e in np.eye(AUGMENTED_STATE_SIZE)
        ]
    )
    np.testing.assert_allclose(H, numeric, atol=1e-8)


def test_clone_is_correlated_copy_and_constant_through_predict():
    ekf = ImuGnssEkf(EkfConfig())
    rng = np.random.default_rng(1)
    state = rng.normal(size=STATE_SIZE)
    A = rng.normal(size=(STATE_SIZE, STATE_SIZE))
    covariance = A @ A.T + np.eye(STATE_SIZE)

    aug_state, aug_cov = ekf.clone_pose(state, covariance)
    pose = [IDX_X, IDX_Y, IDX_YAW]
    np.testing.assert_array_equal(aug_state[[IDX_CLONE_X, IDX_CLONE_Y, IDX_CLONE_YAW]], state[pose])
    np.testing.assert_array_equal(aug_cov[STATE_SIZE:, STATE_SIZE:], covariance[np.ix_(pose, pose)])
    np.testing.assert_array_equal(aug_cov[STATE_SIZE:, :STATE_SIZE], covariance[pose, :])

    imu, dt = np.array([0.3, -0.1, 0.05]), 0.01
    pred_state, pred_cov = ekf.predict(state, covariance, imu, dt)
    pred_aug_state, pred_aug_cov = ekf.predict(aug_state, aug_cov, imu, dt)
    np.testing.assert_allclose(pred_aug_state[:STATE_SIZE], pred_state)
    np.testing.assert_allclose(pred_aug_cov[:STATE_SIZE, :STATE_SIZE], pred_cov)
    np.testing.assert_array_equal(pred_aug_state[STATE_SIZE:], aug_state[STATE_SIZE:])
    np.testing.assert_array_equal(pred_aug_cov[STATE_SIZE:, STATE_SIZE:], aug_cov[STATE_SIZE:, STATE_SIZE:])


def test_run_filter_without_vo_has_no_vo_updates():
    config = SimulatorConfig(total_time=5.0, dt=0.1)
    result = _run(simulate_scenario(config), config, use_vo=False)
    assert np.isnan(result["vo_innovations"]).all()


def _gnss_denied_circle():
    # GNSS only at t=0: absolute yaw is unobservable, VO bounds the drift of everything else.
    config = SimulatorConfig(scenario="circle", dt=0.01, total_time=40.0, gnss_period=1.0, gnss_dropout_probability=1.0)
    return config, simulate_scenario(config)


def test_vo_bounds_drift_without_gnss():
    config, sim = _gnss_denied_circle()
    truth = sim["truth_states"]
    errors = {}
    for use_vo in (False, True):
        estimates = _run(sim, config, use_vo)["state_estimates"]
        errors[use_vo] = np.linalg.norm(estimates[-1, [IDX_X, IDX_Y]] - truth[-1, [IDX_X, IDX_Y]])
    assert errors[True] < 0.2 * errors[False]


@pytest.mark.xfail(
    strict=True,
    reason="Known EKF-VIO inconsistency: Jacobians at changing estimates give spurious yaw information "
    "(truth-linearized yaw sigma stays at the prior); fix with first-estimates Jacobians",
)
def test_yaw_uncertainty_stays_near_prior_without_gnss():
    config, sim = _gnss_denied_circle()
    covariances = _run(sim, config, use_vo=True)["covariances"]
    assert np.sqrt(covariances[-1, IDX_YAW, IDX_YAW]) > 0.5 * EkfConfig().initial_yaw_std_rad


@pytest.mark.parametrize("dropout", [0.0, 0.3])
def test_vo_innovations_are_consistent(dropout):
    config = SimulatorConfig(scenario="demo2", dt=0.01, total_time=60.0, vo_dropout_probability=dropout)
    sim = simulate_scenario(config)
    result = _run(sim, config, use_vo=True)
    innovations = result["vo_innovations"]
    updated = ~np.isnan(innovations[:, 0])
    assert np.count_nonzero(updated) == len(sim["vo_to_index"])
    nis = [v @ np.linalg.solve(S, v) for v, S in zip(innovations[updated], result["vo_innovation_covariances"][updated], strict=True)]
    assert 2.5 < np.mean(nis) < 3.6  # chi-square(3) mean is 3
    assert np.isfinite(result["covariances"]).all()


def test_run_filter_rejects_overlapping_vo_deltas():
    config = SimulatorConfig(total_time=5.0, dt=0.1)
    sim = simulate_scenario(config)
    with pytest.raises(ValueError, match="overlap"):
        _run(sim, config, use_vo=True, vo_from_index=sim["vo_from_index"] - np.r_[0, np.ones(len(sim["vo_from_index"]) - 1, int)])
