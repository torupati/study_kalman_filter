import matplotlib.pyplot as plt
import numpy as np
import pytest

from imu_gnss_ekf_training import EkfConfig, SimulatorConfig, run_filter, simulate_scenario
from imu_gnss_ekf_training.animation import NavAnimator, frame_indices
from imu_gnss_ekf_training.ekf import IDX_X, IDX_Y
from imu_gnss_ekf_training.ellipse import chi2_2dof_scale, covariance_ellipse, mahalanobis_squared, normal_1d_scale
from imu_gnss_ekf_training.navlog import NavLog


def make_log(total_time=6.0, dropout=0.3, with_truth=True) -> NavLog:
    simulator_config = SimulatorConfig(total_time=total_time, dt=0.1, gnss_period=0.5, gnss_dropout_probability=dropout, seed=3)
    scenario = simulate_scenario(simulator_config)
    config = EkfConfig()
    result = run_filter(scenario["imu_measurements"], scenario["gnss_measurements"], scenario["gnss_available"], scenario["dt"], config)
    return NavLog.from_run(
        time=scenario["time"],
        result=result,
        gnss_measurements=scenario["gnss_measurements"],
        gnss_available=scenario["gnss_available"],
        config=config,
        truth_states=scenario["truth_states"] if with_truth else None,
        imu_measurements=scenario["imu_measurements"],
        metadata={"source": "test"},
    )


# ---------------------------------------------------------------- ellipse


def test_chi2_scale_matches_known_quantiles():
    assert chi2_2dof_scale(0.95) == pytest.approx(2.447747, abs=1e-6)
    assert chi2_2dof_scale(1.0 - np.exp(-0.5)) == pytest.approx(1.0)  # "1-sigma" ellipse holds ~39.3%
    assert normal_1d_scale(0.95) == pytest.approx(1.959964, abs=1e-6)


def test_axis_aligned_ellipse():
    k = chi2_2dof_scale(0.95)
    width, height, angle = covariance_ellipse(np.diag([1.0, 9.0]), 0.95)
    assert width == pytest.approx(2 * k * 3.0)
    assert height == pytest.approx(2 * k * 1.0)
    assert abs(np.sin(np.deg2rad(angle))) == pytest.approx(1.0)  # major axis along y


def test_rotated_ellipse_45_degrees():
    rotation = np.array([[np.cos(np.pi / 4), -np.sin(np.pi / 4)], [np.sin(np.pi / 4), np.cos(np.pi / 4)]])
    covariance = rotation @ np.diag([4.0, 1.0]) @ rotation.T
    width, height, angle = covariance_ellipse(covariance, 0.95)
    k = chi2_2dof_scale(0.95)
    assert width == pytest.approx(2 * k * 2.0)
    assert height == pytest.approx(2 * k * 1.0)
    assert np.tan(np.deg2rad(angle)) == pytest.approx(1.0)


def test_ellipse_is_vectorized_and_handles_degenerate_and_nan():
    covariances = np.stack([np.diag([1.0, 1.0]), np.zeros((2, 2)), np.full((2, 2), np.nan)])
    width, height, _ = covariance_ellipse(covariances)
    assert width.shape == (3,)
    assert width[1] == 0.0 and height[1] == 0.0
    assert np.isnan(width[2]) and np.isnan(height[2])


def test_mahalanobis_squared():
    assert mahalanobis_squared(np.array([2.0, 0.0]), np.diag([4.0, 1.0])) == pytest.approx(1.0)
    assert np.isnan(mahalanobis_squared(np.array([np.nan, 0.0]), np.eye(2)))


# ---------------------------------------------------------------- run_filter / NavLog


def test_run_filter_prior_and_innovation_outputs():
    log = make_log()
    available = log.gnss_available
    # Without a GNSS update the posterior is just the prior.
    np.testing.assert_array_equal(log.state_estimates[~available], log.prior_state_estimates[~available])
    np.testing.assert_array_equal(log.covariances[~available], log.prior_covariances[~available])
    assert np.isnan(log.innovations[~available]).all()
    assert np.isfinite(log.innovations[available]).all()
    # A GNSS update never increases position uncertainty.
    posterior_trace = np.trace(log.covariances[available, 0:2, 0:2], axis1=1, axis2=2)
    prior_trace = np.trace(log.prior_covariances[available, 0:2, 0:2], axis1=1, axis2=2)
    assert np.all(posterior_trace <= prior_trace)


def test_navlog_round_trip(tmp_path):
    log = make_log()
    loaded = NavLog.load(log.save(tmp_path / "nav_log.npz"))
    for name in ("time", "state_estimates", "covariances", "prior_covariances", "gnss_measurements", "innovations", "truth_states"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(log, name))
    np.testing.assert_array_equal(loaded.gnss_available, log.gnss_available)
    assert loaded.metadata["source"] == "test"
    assert loaded.metadata["state_names"][IDX_X] == "x"
    assert loaded.metadata["ekf_config"]["gnss_position_std"] == EkfConfig().gnss_position_std


def test_navlog_without_truth_round_trip(tmp_path):
    loaded = NavLog.load(make_log(with_truth=False).save(tmp_path / "nav_log.npz"))
    assert loaded.truth_states is None
    assert not loaded.has_truth


def test_navlog_normalizes_missing_gnss_to_nan():
    log = make_log()
    zero_filled = np.nan_to_num(log.gnss_measurements)
    rebuilt = NavLog(**{**log.__dict__, "gnss_measurements": zero_filled})
    assert np.isnan(rebuilt.gnss_measurements[~rebuilt.gnss_available]).all()


def test_navlog_rejects_bad_shapes_and_versions(tmp_path):
    log = make_log()
    with pytest.raises(ValueError, match="covariances"):
        NavLog(**{**log.__dict__, "covariances": log.covariances[:, :4, :4]})

    path = tmp_path / "old.npz"
    np.savez(path, metadata_json=np.array('{"schema_version": 999}'), time=log.time)
    with pytest.raises(ValueError, match="schema_version"):
        NavLog.load(path)


# ---------------------------------------------------------------- animation


def test_frame_indices_pace_and_bounds():
    time = np.arange(0.0, 10.0 + 1e-9, 0.1)
    indices = frame_indices(time, fps=10, speed=2.0)
    assert indices[0] == 0 and indices[-1] == len(time) - 1
    np.testing.assert_allclose(np.diff(time[indices]), 0.2, atol=1e-9)

    repeated = frame_indices(time, fps=30, speed=1.0, start_time=2.0, end_time=3.0)
    assert time[repeated[0]] == pytest.approx(2.0) and time[repeated[-1]] == pytest.approx(3.0)
    assert np.all(np.diff(repeated) >= 0)


@pytest.mark.parametrize("with_truth, follow", [(True, None), (False, 10.0)])
def test_animator_updates_every_frame(with_truth, follow):
    log = make_log(with_truth=with_truth)
    animator = NavAnimator(log, fps=10, speed=2.0, trail_seconds=2.0, follow_half_width=follow)
    try:
        for frame in range(len(animator.frame_indices)):
            animator.update(frame)
        last = animator.frame_indices[-1]
        center = animator.post_ellipse.get_center()
        assert center == pytest.approx(tuple(log.state_estimates[last, [IDX_X, IDX_Y]]))
        assert animator.post_ellipse.get_width() == pytest.approx(animator.post_width[last])
        assert "t    =" in animator.info_text.get_text()
        animator.figure.canvas.draw()
        (x0, x1), (y0, y1) = animator.ax.get_xlim(), animator.ax.get_ylim()
        if follow:
            # The view is centered on the estimate; the shorter side spans exactly +/- follow.
            assert (0.5 * (x0 + x1), 0.5 * (y0 + y1)) == pytest.approx(center)
            assert min(x1 - x0, y1 - y0) == pytest.approx(2 * follow)
        else:
            # The fixed view contains the whole estimated trajectory.
            estimates = log.state_estimates[:, [IDX_X, IDX_Y]]
            assert x0 <= estimates[:, 0].min() and estimates[:, 0].max() <= x1
            assert y0 <= estimates[:, 1].min() and estimates[:, 1].max() <= y1
    finally:
        plt.close(animator.figure)


def test_animator_hides_gnss_during_outage():
    log = make_log(total_time=6.0, dropout=0.0)
    log.gnss_available[20:] = False  # no GNSS after t = 2 s
    log.gnss_measurements[20:] = np.nan
    animator = NavAnimator(log, fps=10, speed=1.0, gnss_hold_seconds=0.5)
    try:
        animator.update(len(animator.frame_indices) - 1)
        assert not animator.gnss_current.get_visible()
        assert not animator.prior_ellipse.get_visible()
        assert "outage" in animator.info_text.get_text()
    finally:
        plt.close(animator.figure)
