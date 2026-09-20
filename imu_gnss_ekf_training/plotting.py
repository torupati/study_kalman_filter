from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .ekf import IDX_BG, IDX_BAX, IDX_BAY, IDX_VX, IDX_VY, IDX_X, IDX_Y, IDX_YAW, wrap_angle


def create_summary_figure(time: np.ndarray, truth_states: np.ndarray, state_estimates: np.ndarray, gnss_measurements: np.ndarray, gnss_available: np.ndarray):
    """Create the full set of pedagogical EKF plots."""
    position_errors = state_estimates[:, [IDX_X, IDX_Y]] - truth_states[:, [IDX_X, IDX_Y]]
    velocity_errors = state_estimates[:, [IDX_VX, IDX_VY]] - truth_states[:, [IDX_VX, IDX_VY]]
    yaw_error = wrap_angle(state_estimates[:, IDX_YAW] - truth_states[:, IDX_YAW])

    figure, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)

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

    axes[0, 1].plot(time, position_errors[:, 0], label="x error")
    axes[0, 1].plot(time, position_errors[:, 1], label="y error")
    axes[0, 1].set_title("Position errors")
    axes[0, 1].set_xlabel("time [s]")
    axes[0, 1].set_ylabel("error [m]")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(time, velocity_errors[:, 0], label="vx error")
    axes[1, 0].plot(time, velocity_errors[:, 1], label="vy error")
    axes[1, 0].set_title("Velocity errors")
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_ylabel("error [m/s]")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(time, np.rad2deg(yaw_error))
    axes[1, 1].set_title("Yaw error")
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].set_ylabel("error [deg]")
    axes[1, 1].grid(True)

    axes[2, 0].plot(time, truth_states[:, IDX_BAX], label="true bax")
    axes[2, 0].plot(time, state_estimates[:, IDX_BAX], label="estimated bax")
    axes[2, 0].plot(time, truth_states[:, IDX_BAY], label="true bay")
    axes[2, 0].plot(time, state_estimates[:, IDX_BAY], label="estimated bay")
    axes[2, 0].set_title("Accelerometer biases")
    axes[2, 0].set_xlabel("time [s]")
    axes[2, 0].set_ylabel("bias [m/s²]")
    axes[2, 0].legend(ncol=2)
    axes[2, 0].grid(True)

    axes[2, 1].plot(time, truth_states[:, IDX_BG], label="true bg")
    axes[2, 1].plot(time, state_estimates[:, IDX_BG], label="estimated bg")
    axes[2, 1].set_title("Gyro bias")
    axes[2, 1].set_xlabel("time [s]")
    axes[2, 1].set_ylabel("bias [rad/s]")
    axes[2, 1].legend()
    axes[2, 1].grid(True)

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
