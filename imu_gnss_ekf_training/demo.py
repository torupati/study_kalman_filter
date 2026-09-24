from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .ekf import IDX_BAX, IDX_BAY, IDX_BG, IDX_VX, IDX_VY, IDX_X, IDX_Y, IDX_YAW, EkfConfig, run_filter
from .plotting import create_state_timeseries_figure, create_summary_figure, save_figure
from .simulator import SimulatorConfig, simulate_scenario


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the 2D IMU/GNSS EKF training demo.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/imu_gnss_ekf_training"))
    parser.add_argument("--total-time", type=float, default=60.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--gnss-period", type=float, default=0.5)
    parser.add_argument("--gnss-dropout", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def summarize_errors(truth_states: np.ndarray, state_estimates: np.ndarray) -> dict[str, float]:
    position_rmse = np.sqrt(
        np.mean(np.sum((state_estimates[:, [IDX_X, IDX_Y]] - truth_states[:, [IDX_X, IDX_Y]]) ** 2, axis=1))
    )
    velocity_rmse = np.sqrt(
        np.mean(np.sum((state_estimates[:, [IDX_VX, IDX_VY]] - truth_states[:, [IDX_VX, IDX_VY]]) ** 2, axis=1))
    )
    yaw_rmse_deg = np.rad2deg(
        np.sqrt(np.mean(np.arctan2(np.sin(state_estimates[:, IDX_YAW] - truth_states[:, IDX_YAW]),
                                   np.cos(state_estimates[:, IDX_YAW] - truth_states[:, IDX_YAW])) ** 2))
    )
    accel_bias_rmse = np.sqrt(
        np.mean(np.sum((state_estimates[:, [IDX_BAX, IDX_BAY]] - truth_states[:, [IDX_BAX, IDX_BAY]]) ** 2, axis=1))
    )
    gyro_bias_rmse = np.sqrt(np.mean((state_estimates[:, IDX_BG] - truth_states[:, IDX_BG]) ** 2))
    return {
        "position_rmse_m": float(position_rmse),
        "velocity_rmse_mps": float(velocity_rmse),
        "yaw_rmse_deg": float(yaw_rmse_deg),
        "accel_bias_rmse": float(accel_bias_rmse),
        "gyro_bias_rmse": float(gyro_bias_rmse),
    }


def main() -> None:
    args = build_argument_parser().parse_args()
    simulator_config = SimulatorConfig(
        total_time=args.total_time,
        dt=args.dt,
        gnss_period=args.gnss_period,
        gnss_dropout_probability=args.gnss_dropout,
        seed=args.seed,
    )
    scenario = simulate_scenario(simulator_config)
    result = run_filter(
        imu_measurements=np.asarray(scenario["imu_measurements"]),
        gnss_measurements=np.asarray(scenario["gnss_measurements"]),
        gnss_available=np.asarray(scenario["gnss_available"]),
        dt=float(scenario["dt"]),
        config=EkfConfig(
            accel_noise_std=simulator_config.accel_noise_std,
            gyro_noise_std=simulator_config.gyro_noise_std,
            accel_bias_walk_std=simulator_config.accel_bias_walk_std,
            gyro_bias_walk_std=simulator_config.gyro_bias_walk_std,
            gnss_position_std=simulator_config.gnss_position_std,
            gnss_velocity_std=simulator_config.gnss_velocity_std,
        ),
    )

    figure = create_summary_figure(
        time=np.asarray(scenario["time"]),
        truth_states=np.asarray(scenario["truth_states"]),
        state_estimates=np.asarray(result["state_estimates"]),
        gnss_measurements=np.asarray(scenario["gnss_measurements"]),
        gnss_available=np.asarray(scenario["gnss_available"]),
    )
    output_path = save_figure(figure, args.output_dir / "ekf_summary.png")
    print(f"saved figure: {output_path}")

    state_figure = create_state_timeseries_figure(
        time=np.asarray(scenario["time"]),
        state_estimates=np.asarray(result["state_estimates"]),
        covariances=np.asarray(result["covariances"]),
        gnss_measurements=np.asarray(scenario["gnss_measurements"]),
        gnss_available=np.asarray(scenario["gnss_available"]),
    )
    state_output_path = save_figure(state_figure, args.output_dir / "ekf_state_timeseries.png")
    print(f"saved figure: {state_output_path}")

    for key, value in summarize_errors(np.asarray(scenario["truth_states"]), np.asarray(result["state_estimates"])).items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
