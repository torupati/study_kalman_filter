from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .ekf import IDX_BAX, IDX_BAY, IDX_BG, IDX_VX, IDX_VY, IDX_X, IDX_Y, IDX_YAW

GRAVITY_MPS2 = 9.80665


def _disable_axis(ax, message: str) -> None:
    """Blank out an axis and show a placeholder message (used when truth is unavailable)."""
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, fontsize=10, color="gray")
    ax.set_xticks([])
    ax.set_yticks([])


def create_summary_figure(
    time: np.ndarray,
    state_estimates: np.ndarray,
    gnss_measurements: np.ndarray,
    gnss_available: np.ndarray,
    truth_states: np.ndarray | None = None,
):
    """Create the full set of pedagogical EKF plots.

    `truth_states` is optional: when omitted (e.g. running the EKF on recorded
    sensor data rather than a simulation), truth-dependent panels are skipped.
    """
    has_truth = truth_states is not None

    figure, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)

    if has_truth:
        axes[0, 0].plot(truth_states[:, IDX_X], truth_states[:, IDX_Y], label="truth", linewidth=2.0)
    axes[0, 0].plot(state_estimates[:, IDX_X], state_estimates[:, IDX_Y], label="ekf", linewidth=1.5)
    gnss_xy = gnss_measurements[gnss_available, :2]
    axes[0, 0].scatter(gnss_xy[:, 0], gnss_xy[:, 1], s=10, alpha=0.35, label="gnss")
    axes[0, 0].set_title("Trajectory")
    axes[0, 0].set_xlabel("x [m]")
    axes[0, 0].set_ylabel("y [m]")
    axes[0, 0].axis("equal")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    if has_truth:
        position_errors = state_estimates[:, [IDX_X, IDX_Y]] - truth_states[:, [IDX_X, IDX_Y]]
        axes[0, 1].plot(time, position_errors[:, 0], label="x error")
        axes[0, 1].plot(time, position_errors[:, 1], label="y error")
        axes[0, 1].set_title("Position errors")
        axes[0, 1].set_xlabel("time [s]")
        axes[0, 1].set_ylabel("error [m]")
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    else:
        axes[0, 1].set_title("Position errors")
        _disable_axis(axes[0, 1], "no ground truth available")

    if has_truth:
        velocity_errors = state_estimates[:, [IDX_VX, IDX_VY]] - truth_states[:, [IDX_VX, IDX_VY]]
        axes[1, 0].plot(time, velocity_errors[:, 0], label="vx error")
        axes[1, 0].plot(time, velocity_errors[:, 1], label="vy error")
        axes[1, 0].set_title("Velocity errors")
        axes[1, 0].set_xlabel("time [s]")
        axes[1, 0].set_ylabel("error [m/s]")
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    else:
        axes[1, 0].set_title("Velocity errors")
        _disable_axis(axes[1, 0], "no ground truth available")

    if has_truth:
        yaw_error = np.arctan2(np.sin(state_estimates[:, IDX_YAW] - truth_states[:, IDX_YAW]),
                               np.cos(state_estimates[:, IDX_YAW] - truth_states[:, IDX_YAW]))
        axes[1, 1].plot(time, np.rad2deg(yaw_error))
        axes[1, 1].set_title("Yaw error")
        axes[1, 1].set_xlabel("time [s]")
        axes[1, 1].set_ylabel("error [deg]")
        axes[1, 1].grid(True)
    else:
        axes[1, 1].set_title("Yaw error")
        _disable_axis(axes[1, 1], "no ground truth available")

    if has_truth:
        axes[2, 0].plot(time, truth_states[:, IDX_BAX], label="true bax")
    axes[2, 0].plot(time, state_estimates[:, IDX_BAX], label="estimated bax")
    if has_truth:
        axes[2, 0].plot(time, truth_states[:, IDX_BAY], label="true bay")
    axes[2, 0].plot(time, state_estimates[:, IDX_BAY], label="estimated bay")
    axes[2, 0].set_title("Accelerometer biases")
    axes[2, 0].set_xlabel("time [s]")
    axes[2, 0].set_ylabel("bias [m/s²]")
    axes[2, 0].legend(ncol=2)
    axes[2, 0].grid(True)

    if has_truth:
        axes[2, 1].plot(time, truth_states[:, IDX_BG], label="true bg")
    axes[2, 1].plot(time, state_estimates[:, IDX_BG], label="estimated bg")
    axes[2, 1].set_title("Gyro bias")
    axes[2, 1].set_xlabel("time [s]")
    axes[2, 1].set_ylabel("bias [rad/s]")
    axes[2, 1].legend()
    axes[2, 1].grid(True)

    return figure


def create_state_timeseries_figure(
    time: np.ndarray,
    state_estimates: np.ndarray,
    covariances: np.ndarray,
    gnss_measurements: np.ndarray,
    gnss_available: np.ndarray,
):
    """Per-state time series (x, y, vx, vy, yaw) with ±1 std bands and GNSS observations."""
    std = np.sqrt(np.diagonal(covariances, axis1=1, axis2=2))
    gnss_time = time[gnss_available]
    gnss_xy = gnss_measurements[gnss_available, :2]
    gnss_velocity = gnss_measurements[gnss_available, 2:4]

    figure, axes = plt.subplots(5, 1, figsize=(11, 15), sharex=True, constrained_layout=True)
    figure.suptitle("EKF state estimates with uncertainty")

    axes[0].plot(time, state_estimates[:, IDX_X], label="ekf", color="tab:blue")
    axes[0].fill_between(
        time,
        state_estimates[:, IDX_X] - std[:, IDX_X],
        state_estimates[:, IDX_X] + std[:, IDX_X],
        alpha=0.25, color="tab:blue", label="±1 std",
    )
    axes[0].scatter(gnss_time, gnss_xy[:, 0], s=10, alpha=0.6, color="tab:red", label="gnss")
    axes[0].set_ylabel("x [m]")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(time, state_estimates[:, IDX_Y], label="ekf", color="tab:blue")
    axes[1].fill_between(
        time,
        state_estimates[:, IDX_Y] - std[:, IDX_Y],
        state_estimates[:, IDX_Y] + std[:, IDX_Y],
        alpha=0.25, color="tab:blue", label="±1 std",
    )
    axes[1].scatter(gnss_time, gnss_xy[:, 1], s=10, alpha=0.6, color="tab:red", label="gnss")
    axes[1].set_ylabel("y [m]")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(time, state_estimates[:, IDX_VX], label="ekf", color="tab:blue")
    axes[2].fill_between(
        time,
        state_estimates[:, IDX_VX] - std[:, IDX_VX],
        state_estimates[:, IDX_VX] + std[:, IDX_VX],
        alpha=0.25, color="tab:blue", label="±1 std",
    )
    axes[2].scatter(gnss_time, gnss_velocity[:, 0], s=10, alpha=0.6, color="tab:red", label="gnss")
    axes[2].set_ylabel("vx [m/s]")
    axes[2].legend()
    axes[2].grid(True)

    axes[3].plot(time, state_estimates[:, IDX_VY], label="ekf", color="tab:blue")
    axes[3].fill_between(
        time,
        state_estimates[:, IDX_VY] - std[:, IDX_VY],
        state_estimates[:, IDX_VY] + std[:, IDX_VY],
        alpha=0.25, color="tab:blue", label="±1 std",
    )
    axes[3].scatter(gnss_time, gnss_velocity[:, 1], s=10, alpha=0.6, color="tab:red", label="gnss")
    axes[3].set_ylabel("vy [m/s]")
    axes[3].legend()
    axes[3].grid(True)

    yaw_deg = np.rad2deg(state_estimates[:, IDX_YAW])
    yaw_std_deg = np.rad2deg(std[:, IDX_YAW])
    axes[4].plot(time, yaw_deg, label="ekf", color="tab:blue")
    axes[4].fill_between(time, yaw_deg - yaw_std_deg, yaw_deg + yaw_std_deg, alpha=0.25, color="tab:blue", label="±1 std")
    axes[4].set_ylabel("yaw [deg]")
    axes[4].set_xlabel("time [s]")
    axes[4].legend()
    axes[4].grid(True)

    return figure


def create_trajectory_figure(
    state_estimates: np.ndarray,
    gnss_measurements: np.ndarray,
    gnss_available: np.ndarray,
    truth_states: np.ndarray | None = None,
):
    """EKF trajectory (x vs y), plotted as markers only, with GNSS and optional truth overlaid."""
    figure, ax = plt.subplots(figsize=(7, 6))

    if truth_states is not None:
        ax.plot(
            truth_states[:, IDX_X], truth_states[:, IDX_Y],
            linestyle="none", marker="x", markersize=4, color="tab:orange", alpha=0.6, label="truth",
        )
    gnss_xy = gnss_measurements[gnss_available, :2]
    ax.plot(
        gnss_xy[:, 0], gnss_xy[:, 1],
        linestyle="none", marker="o", markersize=5, color="tab:red", alpha=0.5, label="gnss",
    )
    ax.plot(
        state_estimates[:, IDX_X], state_estimates[:, IDX_Y],
        linestyle="none", marker=".", markersize=3, color="tab:blue", label="ekf",
    )
    ax.set_title("EKF trajectory")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.legend()
    ax.grid(True)
    return figure


def create_output_timeseries_figure(
    time: np.ndarray,
    state_estimates: np.ndarray,
    gnss_measurements: np.ndarray,
    gnss_available: np.ndarray,
    truth_states: np.ndarray | None = None,
):
    """EKF output (position, velocity, yaw, biases) as marker-only time series, one axis each.

    GNSS observations are overlaid on the position/velocity panels. `truth_states` is
    optional: when given, true values are overlaid on every panel.
    """
    ekf_kwargs = dict(linestyle="none", marker=".", markersize=3, color="tab:blue", label="ekf")
    gnss_kwargs = dict(linestyle="none", marker="o", markersize=5, color="tab:red", alpha=0.5, label="gnss")
    truth_kwargs = dict(linestyle="none", marker="x", markersize=4, color="tab:orange", alpha=0.6, label="truth")

    gnss_time = time[gnss_available]
    gnss_xy = gnss_measurements[gnss_available, :2]
    gnss_velocity = gnss_measurements[gnss_available, 2:4]
    has_truth = truth_states is not None

    figure, axes = plt.subplots(4, 2, figsize=(13, 14), constrained_layout=True)
    figure.suptitle("EKF output")

    if has_truth:
        axes[0, 0].plot(time, truth_states[:, IDX_X], **truth_kwargs)
    axes[0, 0].plot(gnss_time, gnss_xy[:, 0], **gnss_kwargs)
    axes[0, 0].plot(time, state_estimates[:, IDX_X], **ekf_kwargs)
    axes[0, 0].set_title("x")
    axes[0, 0].set_ylabel("x [m]")
    axes[0, 0].legend()

    if has_truth:
        axes[0, 1].plot(time, truth_states[:, IDX_Y], **truth_kwargs)
    axes[0, 1].plot(gnss_time, gnss_xy[:, 1], **gnss_kwargs)
    axes[0, 1].plot(time, state_estimates[:, IDX_Y], **ekf_kwargs)
    axes[0, 1].set_title("y")
    axes[0, 1].set_ylabel("y [m]")
    axes[0, 1].legend()

    if has_truth:
        axes[1, 0].plot(time, truth_states[:, IDX_VX], **truth_kwargs)
    axes[1, 0].plot(gnss_time, gnss_velocity[:, 0], **gnss_kwargs)
    axes[1, 0].plot(time, state_estimates[:, IDX_VX], **ekf_kwargs)
    axes[1, 0].set_title("vx")
    axes[1, 0].set_ylabel("vx [m/s]")
    axes[1, 0].legend()

    if has_truth:
        axes[1, 1].plot(time, truth_states[:, IDX_VY], **truth_kwargs)
    axes[1, 1].plot(gnss_time, gnss_velocity[:, 1], **gnss_kwargs)
    axes[1, 1].plot(time, state_estimates[:, IDX_VY], **ekf_kwargs)
    axes[1, 1].set_title("vy")
    axes[1, 1].set_ylabel("vy [m/s]")
    axes[1, 1].legend()

    if has_truth:
        axes[2, 0].plot(time, np.rad2deg(truth_states[:, IDX_YAW]), **truth_kwargs)
    axes[2, 0].plot(time, np.rad2deg(state_estimates[:, IDX_YAW]), **ekf_kwargs)
    axes[2, 0].set_title("yaw")
    axes[2, 0].set_ylabel("yaw [deg]")
    axes[2, 0].legend()

    if has_truth:
        axes[2, 1].plot(time, truth_states[:, IDX_BAX] / GRAVITY_MPS2, **truth_kwargs)
    axes[2, 1].plot(time, state_estimates[:, IDX_BAX] / GRAVITY_MPS2, **ekf_kwargs)
    axes[2, 1].set_title("accel_x bias")
    axes[2, 1].set_ylabel("bias [G]")
    axes[2, 1].legend()

    if has_truth:
        axes[3, 0].plot(time, truth_states[:, IDX_BAY] / GRAVITY_MPS2, **truth_kwargs)
    axes[3, 0].plot(time, state_estimates[:, IDX_BAY] / GRAVITY_MPS2, **ekf_kwargs)
    axes[3, 0].set_title("accel_y bias")
    axes[3, 0].set_ylabel("bias [G]")
    axes[3, 0].set_xlabel("time [s]")
    axes[3, 0].legend()

    if has_truth:
        axes[3, 1].plot(time, np.rad2deg(truth_states[:, IDX_BG]), **truth_kwargs)
    axes[3, 1].plot(time, np.rad2deg(state_estimates[:, IDX_BG]), **ekf_kwargs)
    axes[3, 1].set_title("gyro bias")
    axes[3, 1].set_ylabel("bias [deg/s]")
    axes[3, 1].set_xlabel("time [s]")
    axes[3, 1].legend()

    for ax in axes.flat:
        ax.grid(True)

    return figure


def create_experiment_figure(cases: list[dict[str, np.ndarray | str]]):
    """Compare trajectory and position error across multiple scenarios."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for case in cases:
        label = str(case["label"])
        truth_states = np.asarray(case["truth_states"])
        state_estimates = np.asarray(case["state_estimates"])
        time = np.asarray(case["time"])
        position_error_norm = np.linalg.norm(
            state_estimates[:, [IDX_X, IDX_Y]] - truth_states[:, [IDX_X, IDX_Y]], axis=1
        )

        axes[0].plot(truth_states[:, IDX_X], truth_states[:, IDX_Y], linestyle="--", alpha=0.35)
        axes[0].plot(state_estimates[:, IDX_X], state_estimates[:, IDX_Y], label=label)
        axes[1].plot(time, position_error_norm, label=label)

    axes[0].set_title("Experiment trajectories")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].axis("equal")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].set_title("Position error norm")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("|position error| [m]")
    axes[1].legend()
    axes[1].grid(True)
    return figure


def save_figure(figure, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    return output_path
