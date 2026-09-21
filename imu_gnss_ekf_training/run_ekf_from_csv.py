"""CLI: run the EKF on recorded IMU/GNSS CSV files (from `record_sensors.py`) and plot results.

Usage
-----
    uv run python -m imu_gnss_ekf_training.record_sensors --output-dir outputs/imu_gnss_ekf_training
    uv run python -m imu_gnss_ekf_training.run_ekf_from_csv --output-dir outputs/imu_gnss_ekf_training

Optional initial state (position/velocity/yaw; biases always start at zero), as JSON. Mean values:
    {"x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "yaw_deg": 0.0}
Optionally also give the initial-state accuracy (missing keys fall back to the --initial-*-std
CLI defaults below):
    {"cov_pos_xx": 25.0, "cov_pos_yy": 25.0, "cov_pos_xy": 0.0,
     "cov_vel_xx": 4.0, "cov_vel_yy": 4.0, "cov_vel_xy": 0.0, "yaw_std_deg": 45.0}

Optional ground-truth state CSV (from `record_sensors.py --save-true-state`) enables error
panels in ekf_summary_from_csv.png and printed RMSE metrics.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
from pathlib import Path

import numpy as np

from .demo import summarize_errors
from .ekf import STATE_SIZE, IDX_BG, IDX_BAX, IDX_BAY, IDX_VX, IDX_VY, IDX_X, IDX_Y, IDX_YAW, EkfConfig, run_filter
from .plotting import (
    create_output_timeseries_figure,
    create_state_timeseries_figure,
    create_summary_figure,
    create_trajectory_figure,
    save_figure,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the 2D IMU/GNSS EKF on recorded sensor CSV files."
    )
    parser.add_argument("--imu-csv", type=Path, default=Path("outputs/imu_gnss_ekf_training/imu_observations.csv"))
    parser.add_argument("--gnss-csv", type=Path, default=Path("outputs/imu_gnss_ekf_training/gnss_observations.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/imu_gnss_ekf_training"))
    parser.add_argument("--accel-noise-std", type=float, default=EkfConfig().accel_noise_std)
    parser.add_argument("--gyro-noise-std", type=float, default=EkfConfig().gyro_noise_std)
    parser.add_argument("--accel-bias-walk-std", type=float, default=EkfConfig().accel_bias_walk_std)
    parser.add_argument("--gyro-bias-walk-std", type=float, default=EkfConfig().gyro_bias_walk_std)
    parser.add_argument("--gnss-position-std", type=float, default=EkfConfig().gnss_position_std)
    parser.add_argument("--gnss-velocity-std", type=float, default=EkfConfig().gnss_velocity_std)
    parser.add_argument("--initial-position-std", type=float, default=EkfConfig().initial_position_std)
    parser.add_argument("--initial-velocity-std", type=float, default=EkfConfig().initial_velocity_std)
    parser.add_argument("--initial-yaw-std-deg", type=float, default=np.rad2deg(EkfConfig().initial_yaw_std_rad))
    parser.add_argument("--initial-accel-bias-std", type=float, default=EkfConfig().initial_accel_bias_std)
    parser.add_argument("--initial-gyro-bias-std", type=float, default=EkfConfig().initial_gyro_bias_std)
    parser.add_argument(
        "--initial-state-json",
        type=Path,
        default=None,
        help='Optional JSON file with initial {"x", "y", "vx", "vy", "yaw_deg"} and optionally '
        '{"cov_pos_xx", "cov_pos_yy", "cov_pos_xy", "cov_vel_xx", "cov_vel_yy", "cov_vel_xy", "yaw_std_deg"}; '
        "missing keys fall back to the --initial-*-std options above. Biases always start at zero.",
    )
    parser.add_argument(
        "--true-state-csv",
        type=Path,
        default=None,
        help="Optional ground-truth state CSV (from record_sensors.py --save-true-state) to compute errors against.",
    )
    return parser


def load_imu_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read (time_s, accel_x_mss, accel_y_mss, gyro_z_rads) rows into (times, imu_measurements)."""
    times: list[float] = []
    rows: list[list[float]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["time_s"]))
            rows.append([float(row["accel_x_mss"]), float(row["accel_y_mss"]), float(row["gyro_z_rads"])])
    return np.asarray(times, dtype=float), np.asarray(rows, dtype=float)


def load_gnss_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read sparse (time_s, pos_x_m, pos_y_m, vel_x_mps, vel_y_mps) rows into (times, gnss_rows)."""
    times: list[float] = []
    rows: list[list[float]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["time_s"]))
            rows.append([float(row["pos_x_m"]), float(row["pos_y_m"]), float(row["vel_x_mps"]), float(row["vel_y_mps"])])
    return np.asarray(times, dtype=float), np.asarray(rows, dtype=float)


def load_true_state_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read (time, x, y, vx, vy, yaw, ax_bias, ay_bias, gz_bias) rows into (times, truth_states)."""
    times: list[float] = []
    rows: list[list[float]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["time"]))
            state = [0.0] * STATE_SIZE
            state[IDX_X] = float(row["x"])
            state[IDX_Y] = float(row["y"])
            state[IDX_VX] = float(row["vx"])
            state[IDX_VY] = float(row["vy"])
            state[IDX_YAW] = float(row["yaw"])
            state[IDX_BAX] = float(row["ax_bias"])
            state[IDX_BAY] = float(row["ay_bias"])
            state[IDX_BG] = float(row["gz_bias"])
            rows.append(state)
    return np.asarray(times, dtype=float), np.asarray(rows, dtype=float)


def align_truth_to_imu(
    imu_times: np.ndarray, truth_times: np.ndarray, truth_rows: np.ndarray, dt: float
) -> np.ndarray:
    """Snap ground-truth state rows onto the dense IMU time grid, within half a sample period."""
    truth_states = np.full((imu_times.shape[0], STATE_SIZE), np.nan, dtype=float)

    for truth_time, truth_row in zip(truth_times, truth_rows):
        candidate = int(np.searchsorted(imu_times, truth_time))
        best_index = min(
            (index for index in (candidate - 1, candidate) if 0 <= index < imu_times.shape[0]),
            key=lambda index: abs(imu_times[index] - truth_time),
        )
        if abs(imu_times[best_index] - truth_time) <= dt / 2 + 1e-9:
            truth_states[best_index] = truth_row

    return truth_states


def default_initial_covariance(config: EkfConfig) -> np.ndarray:
    """Uncorrelated initial covariance built from `config`'s initial-*-std values."""
    return np.diag(
        [
            config.initial_position_std**2,
            config.initial_position_std**2,
            config.initial_velocity_std**2,
            config.initial_velocity_std**2,
            config.initial_yaw_std_rad**2,
            config.initial_accel_bias_std**2,
            config.initial_accel_bias_std**2,
            config.initial_gyro_bias_std**2,
        ]
    )


def load_initial_state_json(path: Path, config: EkfConfig) -> tuple[np.ndarray, np.ndarray]:
    """Load an initial (x, y, vx, vy, yaw_deg) state and its covariance from JSON.

    Biases always start at zero. Covariance defaults to `config`'s uncorrelated
    initial-*-std values; JSON may override position/velocity/yaw accuracy via
    cov_pos_xx, cov_pos_yy, cov_pos_xy, cov_vel_xx, cov_vel_yy, cov_vel_xy, yaw_std_deg.
    """
    with path.open() as f:
        data = json.load(f)

    state = np.zeros(STATE_SIZE, dtype=float)
    state[IDX_X] = float(data.get("x", 0.0))
    state[IDX_Y] = float(data.get("y", 0.0))
    state[IDX_VX] = float(data.get("vx", 0.0))
    state[IDX_VY] = float(data.get("vy", 0.0))
    state[IDX_YAW] = np.deg2rad(float(data.get("yaw_deg", 0.0)))

    covariance = default_initial_covariance(config)
    covariance[IDX_X, IDX_X] = float(data.get("cov_pos_xx", covariance[IDX_X, IDX_X]))
    covariance[IDX_Y, IDX_Y] = float(data.get("cov_pos_yy", covariance[IDX_Y, IDX_Y]))
    covariance[IDX_X, IDX_Y] = covariance[IDX_Y, IDX_X] = float(data.get("cov_pos_xy", 0.0))
    covariance[IDX_VX, IDX_VX] = float(data.get("cov_vel_xx", covariance[IDX_VX, IDX_VX]))
    covariance[IDX_VY, IDX_VY] = float(data.get("cov_vel_yy", covariance[IDX_VY, IDX_VY]))
    covariance[IDX_VX, IDX_VY] = covariance[IDX_VY, IDX_VX] = float(data.get("cov_vel_xy", 0.0))
    if "yaw_std_deg" in data:
        covariance[IDX_YAW, IDX_YAW] = np.deg2rad(float(data["yaw_std_deg"])) ** 2

    return state, covariance


def align_gnss_to_imu(
    imu_times: np.ndarray, gnss_times: np.ndarray, gnss_rows: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Snap sparse GNSS rows onto the dense IMU time grid, within half a sample period."""
    gnss_measurements = np.zeros((imu_times.shape[0], 4), dtype=float)
    gnss_available = np.zeros(imu_times.shape[0], dtype=bool)

    for gnss_time, gnss_row in zip(gnss_times, gnss_rows):
        candidate = int(np.searchsorted(imu_times, gnss_time))
        best_index = min(
            (index for index in (candidate - 1, candidate) if 0 <= index < imu_times.shape[0]),
            key=lambda index: abs(imu_times[index] - gnss_time),
        )
        if abs(imu_times[best_index] - gnss_time) <= dt / 2 + 1e-9:
            gnss_available[best_index] = True
            gnss_measurements[best_index] = gnss_row

    return gnss_measurements, gnss_available


def save_run_config_json(path: Path, config: EkfConfig, initial_state: np.ndarray, initial_covariance: np.ndarray) -> None:
    """Save the sensor-noise config and initial-state condition actually used, as JSON.

    The "initial_state" block round-trips as an --initial-state-json input for a later run.
    """
    initial_state_data = {
        "x": float(initial_state[IDX_X]),
        "y": float(initial_state[IDX_Y]),
        "vx": float(initial_state[IDX_VX]),
        "vy": float(initial_state[IDX_VY]),
        "yaw_deg": float(np.rad2deg(initial_state[IDX_YAW])),
        "cov_pos_xx": float(initial_covariance[IDX_X, IDX_X]),
        "cov_pos_yy": float(initial_covariance[IDX_Y, IDX_Y]),
        "cov_pos_xy": float(initial_covariance[IDX_X, IDX_Y]),
        "cov_vel_xx": float(initial_covariance[IDX_VX, IDX_VX]),
        "cov_vel_yy": float(initial_covariance[IDX_VY, IDX_VY]),
        "cov_vel_xy": float(initial_covariance[IDX_VX, IDX_VY]),
        "yaw_std_deg": float(np.rad2deg(np.sqrt(initial_covariance[IDX_YAW, IDX_YAW]))),
    }
    run_config = {
        "initial_state": initial_state_data,
        "ekf_config": dataclasses.asdict(config),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(run_config, f, indent=2)


def main() -> None:
    args = build_argument_parser().parse_args()

    imu_times, imu_measurements = load_imu_csv(args.imu_csv)
    gnss_times, gnss_rows = load_gnss_csv(args.gnss_csv)
    dt = float(np.median(np.diff(imu_times)))
    gnss_measurements, gnss_available = align_gnss_to_imu(imu_times, gnss_times, gnss_rows, dt)

    config = EkfConfig(
        accel_noise_std=args.accel_noise_std,
        gyro_noise_std=args.gyro_noise_std,
        accel_bias_walk_std=args.accel_bias_walk_std,
        gyro_bias_walk_std=args.gyro_bias_walk_std,
        gnss_position_std=args.gnss_position_std,
        gnss_velocity_std=args.gnss_velocity_std,
        initial_position_std=args.initial_position_std,
        initial_velocity_std=args.initial_velocity_std,
        initial_yaw_std_rad=np.deg2rad(args.initial_yaw_std_deg),
        initial_accel_bias_std=args.initial_accel_bias_std,
        initial_gyro_bias_std=args.initial_gyro_bias_std,
    )

    if args.initial_state_json is not None:
        initial_state, initial_covariance = load_initial_state_json(args.initial_state_json, config)
    else:
        initial_state = np.zeros(STATE_SIZE, dtype=float)
        initial_covariance = default_initial_covariance(config)

    truth_states = None
    if args.true_state_csv is not None:
        truth_times, truth_rows = load_true_state_csv(args.true_state_csv)
        truth_states = align_truth_to_imu(imu_times, truth_times, truth_rows, dt)

    result = run_filter(
        imu_measurements=imu_measurements,
        gnss_measurements=gnss_measurements,
        gnss_available=gnss_available,
        dt=dt,
        config=config,
        initial_state=initial_state,
        initial_covariance=initial_covariance,
    )

    figure = create_summary_figure(
        time=imu_times,
        state_estimates=np.asarray(result["state_estimates"]),
        gnss_measurements=gnss_measurements,
        gnss_available=gnss_available,
        truth_states=truth_states,
    )
    output_path = save_figure(figure, args.output_dir / "ekf_summary_from_csv.png")

    state_figure = create_state_timeseries_figure(
        time=imu_times,
        state_estimates=np.asarray(result["state_estimates"]),
        covariances=np.asarray(result["covariances"]),
        gnss_measurements=gnss_measurements,
        gnss_available=gnss_available,
    )
    state_output_path = save_figure(state_figure, args.output_dir / "ekf_state_timeseries_from_csv.png")

    trajectory_figure = create_trajectory_figure(
        state_estimates=np.asarray(result["state_estimates"]),
        gnss_measurements=gnss_measurements,
        gnss_available=gnss_available,
        truth_states=truth_states,
    )
    trajectory_output_path = save_figure(trajectory_figure, args.output_dir / "ekf_trajectory_from_csv.png")

    output_timeseries_figure = create_output_timeseries_figure(
        time=imu_times,
        state_estimates=np.asarray(result["state_estimates"]),
        gnss_measurements=gnss_measurements,
        gnss_available=gnss_available,
        truth_states=truth_states,
    )
    output_timeseries_path = save_figure(output_timeseries_figure, args.output_dir / "ekf_output_timeseries_from_csv.png")

    run_config_path = args.output_dir / "run_config.json"
    save_run_config_json(run_config_path, config, initial_state, initial_covariance)

    print(f"IMU samples : {imu_times.shape[0]} (dt={dt:.4f}s)")
    print(f"GNSS fixes  : {int(gnss_available.sum())} / {gnss_times.shape[0]} matched onto IMU grid")
    if args.initial_state_json is not None:
        print(f"initial state loaded from: {args.initial_state_json}")
    print(f"saved run config: {run_config_path}")
    print(f"saved figure: {output_path}")
    print(f"saved figure: {state_output_path}")
    print(f"saved figure: {trajectory_output_path}")
    print(f"saved figure: {output_timeseries_path}")

    if truth_states is not None:
        for key, value in summarize_errors(truth_states, np.asarray(result["state_estimates"])).items():
            print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
