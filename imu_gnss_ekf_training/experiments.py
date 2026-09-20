from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .demo import summarise_errors
from .ekf import EkfConfig, run_filter
from .plotting import create_experiment_figure, save_figure
from .simulator import SimulatorConfig, simulate_scenario


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run EKF training experiments with different GNSS dropout settings.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/imu_gnss_ekf_training"))
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    cases: list[dict[str, np.ndarray | str]] = []
    for label, dropout in [("dense GNSS", 0.0), ("intermittent GNSS", 0.35)]:
        simulator_config = SimulatorConfig(gnss_dropout_probability=dropout, seed=args.seed)
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
        summary = summarise_errors(np.asarray(scenario["truth_states"]), np.asarray(result["state_estimates"]))
        print(f"{label}:")
        for key, value in summary.items():
            print(f"  {key}: {value:.6f}")
        cases.append(
            {
                "label": label,
                "time": np.asarray(scenario["time"]),
                "truth_states": np.asarray(scenario["truth_states"]),
                "state_estimates": np.asarray(result["state_estimates"]),
            }
        )

    figure = create_experiment_figure(cases)
    output_path = save_figure(figure, args.output_dir / "dropout_experiments.png")
    print(f"saved figure: {output_path}")


if __name__ == "__main__":
    main()
