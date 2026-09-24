"""CLI: simulate sensor data, export as CSV, and plot raw observations.

Usage
-----
    uv run python -m imu_gnss_ekf_training.record_sensors demo1
    uv run python -m imu_gnss_ekf_training.record_sensors demo2 --output-dir outputs/sensors --total-time 90

Scenarios
---------
    demo1       Smooth, continuously curving trajectory (the original demo).
    demo2       Accelerate, cruise straight for 30 m, half-circle turn, cruise back, stop.
    line        Straight-line trajectory. Not implemented yet.
    stationary  Vehicle stays put. Not implemented yet.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .ekf import IDX_BAX, IDX_BAY, IDX_BG, IDX_VX, IDX_VY, IDX_X, IDX_Y, IDX_YAW
from .simulator import SimulatorConfig, simulate_scenario

GRAVITY_MPS2 = 9.80665
IMU_SAMPLE_RATE_HZ = 100.0


NOT_YET_IMPLEMENTED_SCENARIOS = ("line", "stationary")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulate and record IMU/GNSS observations as CSV, then plot them."
    )

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-dir", type=Path, default=Path("outputs/imu_gnss_ekf_training"))
    common.add_argument("--total-time", type=float, default=60.0)
    common.add_argument("--dt", type=float, default=1.0 / IMU_SAMPLE_RATE_HZ, help="IMU sample period in seconds (default: 100 Hz).")
    common.add_argument("--gnss-period", type=float, default=0.5)
    common.add_argument("--gnss-dropout", type=float, default=0.15)
    common.add_argument("--seed", type=int, default=7)
    common.add_argument("--no-show", action="store_true", help="Save figures without displaying them.")
    common.add_argument(
        "--save-true-state",
        action="store_true",
        help="Also write the simulator's ground-truth state (position, velocity, yaw, biases) to CSV.",
    )

    subparsers = parser.add_subparsers(dest="scenario", required=True, help="Trajectory scenario to simulate.")
    subparsers.add_parser("demo1", parents=[common], help="Smooth, continuously curving demo trajectory.")
    subparsers.add_parser("demo2", parents=[common], help="Straight 30 m out, half-circle turn, straight back.")
    subparsers.add_parser("line", parents=[common], help="Straight-line trajectory (not implemented yet).")
    subparsers.add_parser("stationary", parents=[common], help="Vehicle stays put (not implemented yet).")
    return parser


def write_imu_csv(path: Path, times: np.ndarray, imu: np.ndarray) -> None:
    """Write IMU observations (accel_x, accel_y, gyro_z) to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "accel_x_mss", "accel_y_mss", "gyro_z_rads"])
        for t, row in zip(times, imu):
            writer.writerow([f"{t:.6f}", f"{row[0]:.8f}", f"{row[1]:.8f}", f"{row[2]:.8f}"])


def write_gnss_csv(path: Path, times: np.ndarray, gnss: np.ndarray, available: np.ndarray) -> None:
    """Write GNSS observations (only rows where GNSS is available) to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "pos_x_m", "pos_y_m", "vel_x_mps", "vel_y_mps"])
        for t, row, avail in zip(times, gnss, available):
            if avail:
                writer.writerow([f"{t:.6f}", f"{row[0]:.8f}", f"{row[1]:.8f}", f"{row[2]:.8f}", f"{row[3]:.8f}"])


def write_true_state_csv(path: Path, times: np.ndarray, truth_states: np.ndarray) -> None:
    """Write ground-truth state (position, velocity, yaw, biases) to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "x", "y", "vx", "vy", "yaw", "ax_bias", "ay_bias", "gz_bias"])
        for t, row in zip(times, truth_states):
            writer.writerow(
                [
                    f"{t:.6f}",
                    f"{row[IDX_X]:.8f}",
                    f"{row[IDX_Y]:.8f}",
                    f"{row[IDX_VX]:.8f}",
                    f"{row[IDX_VY]:.8f}",
                    f"{row[IDX_YAW]:.8f}",
                    f"{row[IDX_BAX]:.8f}",
                    f"{row[IDX_BAY]:.8f}",
                    f"{row[IDX_BG]:.8f}",
                ]
            )


def plot_imu_timeseries(times: np.ndarray, imu: np.ndarray) -> plt.Figure:
    """Figure 1: accel_x, accel_y, gyro_z as separate time-series subplots (G, G, deg/s)."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    fig.suptitle("IMU observations")

    axes[0].plot(times, imu[:, 0] / GRAVITY_MPS2, linewidth=0.8, color="tab:blue")
    axes[0].set_ylabel("accel_x (G)")
    axes[0].grid(True, linewidth=0.4)

    axes[1].plot(times, imu[:, 1] / GRAVITY_MPS2, linewidth=0.8, color="tab:orange")
    axes[1].set_ylabel("accel_y (G)")
    axes[1].grid(True, linewidth=0.4)

    axes[2].plot(times, np.rad2deg(imu[:, 2]), linewidth=0.8, color="tab:green")
    axes[2].set_ylabel("gyro_z (deg/s)")
    axes[2].set_xlabel("time (s)")
    axes[2].grid(True, linewidth=0.4)

    fig.tight_layout()
    return fig


def plot_gnss_position(gnss: np.ndarray, available: np.ndarray) -> plt.Figure:
    """Figure 2: 2-D scatter of GNSS position observations."""
    valid = gnss[available]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(valid[:, 0], valid[:, 1], s=6, alpha=0.7, color="tab:red", label="GNSS position")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("GNSS position observations (2D)")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, linewidth=0.4)
    fig.tight_layout()
    return fig


def main() -> None:
    args = build_argument_parser().parse_args()

    if args.scenario in NOT_YET_IMPLEMENTED_SCENARIOS:
        raise NotImplementedError(f"scenario {args.scenario!r} is not implemented yet.")

    config = SimulatorConfig(
        total_time=args.total_time,
        dt=args.dt,
        gnss_period=args.gnss_period,
        gnss_dropout_probability=args.gnss_dropout,
        seed=args.seed,
        scenario=args.scenario,
    )
    scenario = simulate_scenario(config)

    times: np.ndarray = np.asarray(scenario["time"])
    imu: np.ndarray = np.asarray(scenario["imu_measurements"])
    gnss: np.ndarray = np.asarray(scenario["gnss_measurements"])
    available: np.ndarray = np.asarray(scenario["gnss_available"])

    imu_path = args.output_dir / "imu_observations.csv"
    gnss_path = args.output_dir / "gnss_observations.csv"
    write_imu_csv(imu_path, times, imu)
    write_gnss_csv(gnss_path, times, gnss, available)
    print(f"IMU CSV  : {imu_path}  ({len(times)} rows)")
    print(f"GNSS CSV : {gnss_path}  ({available.sum()} rows)")

    if args.save_true_state:
        truth_states: np.ndarray = np.asarray(scenario["truth_states"])
        true_state_path = args.output_dir / "true_state.csv"
        write_true_state_csv(true_state_path, times, truth_states)
        print(f"True state CSV : {true_state_path}  ({len(times)} rows)")

    fig_imu = plot_imu_timeseries(times, imu)
    fig_pos = plot_gnss_position(gnss, available)

    imu_fig_path = args.output_dir / "imu_timeseries.png"
    pos_fig_path = args.output_dir / "gnss_position_2d.png"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig_imu.savefig(imu_fig_path, dpi=150)
    fig_pos.savefig(pos_fig_path, dpi=150)
    print(f"IMU plot : {imu_fig_path}")
    print(f"Pos plot : {pos_fig_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
