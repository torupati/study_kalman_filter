from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np

from .alignment import align_trajectory
from .ekf import IDX_BAX, IDX_BAY, IDX_BG, IDX_VX, IDX_VY, IDX_X, IDX_Y, IDX_YAW, EkfConfig, run_filter
from .navlog import NavLog
from .plotting import (
    create_bias_timeseries_figure,
    create_frame_alignment_figure,
    create_state_timeseries_figure,
    create_summary_figure,
    save_figure,
)
from .run_conditions import save_run_conditions
from .simulator import SCENARIOS, SimulatorConfig, simulate_scenario


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the 2D IMU/GNSS EKF training demo.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/imu_gnss_ekf_training"))
    parser.add_argument("--scenario", choices=SCENARIOS, default="demo1", help="Truth trajectory to simulate")
    parser.add_argument("--total-time", type=float, default=60.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--gnss-period", type=float, default=0.5)
    parser.add_argument("--gnss-dropout", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--accel-noise-density", type=float, default=SimulatorConfig().accel_noise_density,
        help="Simulated accelerometer white noise density [m/s^2/sqrt(Hz)]; per-sample std = density / sqrt(dt)",
    )
    parser.add_argument(
        "--gyro-noise-density", type=float, default=SimulatorConfig().gyro_noise_density,
        help="Simulated gyro white noise density [rad/s/sqrt(Hz)]; per-sample std = density / sqrt(dt)",
    )
    parser.add_argument(
        "--forward-back-distance", type=float, default=SimulatorConfig().forward_back_distance,
        help="One-way travel distance of the forward_back scenario [m]",
    )
    parser.add_argument(
        "--initial-accel-bias", type=float, nargs=2, metavar=("BX", "BY"), default=SimulatorConfig().initial_accel_bias,
        help="True accelerometer bias at t=0, body frame [m/s^2]",
    )
    parser.add_argument(
        "--initial-gyro-bias-dps", type=float, default=float(np.rad2deg(SimulatorConfig().initial_gyro_bias)),
        help="True gyro bias at t=0 [deg/s]",
    )
    parser.add_argument(
        "--initial-pose", type=float, nargs=3, metavar=("X", "Y", "YAW_DEG"), default=None,
        help="True start pose in the world frame [m, m, deg] (default: 0 0 0). The EKF always starts at (0, 0, 0)",
    )
    nav = parser.add_argument_group("navigation", "Which simulated sensors the EKF uses and how it is initialized")
    nav.add_argument(
        "--no-gnss-update", action="store_true",
        help="Simulate GNSS but do not fuse it: the EKF navigates in its own frame anchored at the start "
        "(use with --use-vo); also writes ekf_frame_alignment.png",
    )
    nav.add_argument(
        "--ekf-initial-position-std", type=float, default=None,
        help="EKF initial position std [m] (default: EkfConfig value, or 0.01 with --no-gnss-update)",
    )
    nav.add_argument(
        "--ekf-initial-yaw-std-deg", type=float, default=None,
        help="EKF initial yaw std [deg] (default: EkfConfig value, or 0.1 with --no-gnss-update)",
    )
    vo = parser.add_argument_group("visual odometry", "Simulated VO/SLAM delta poses (see doc/visual_odometry.md)")
    vo.add_argument("--use-vo", action="store_true", help="Fuse VO delta poses in the EKF (stochastic cloning)")
    vo.add_argument("--vo-period", type=float, default=SimulatorConfig().vo_period, help="VO frame period [s]")
    vo.add_argument(
        "--vo-translation-std", type=float, default=SimulatorConfig().vo_translation_std,
        help="VO translation noise floor per frame [m]",
    )
    vo.add_argument(
        "--vo-translation-std-per-m", type=float, default=SimulatorConfig().vo_translation_std_per_m,
        help="Extra VO translation noise per meter moved [m/m]",
    )
    vo.add_argument(
        "--vo-yaw-std-deg", type=float, default=float(np.rad2deg(SimulatorConfig().vo_yaw_std)),
        help="VO yaw noise floor per frame [deg]",
    )
    vo.add_argument(
        "--vo-yaw-std-per-rad", type=float, default=SimulatorConfig().vo_yaw_std_per_rad,
        help="Extra VO yaw noise per radian turned [rad/rad]",
    )
    vo.add_argument(
        "--vo-dropout", type=float, default=SimulatorConfig().vo_dropout_probability,
        help="Probability that a VO frame loses tracking",
    )
    parser.add_argument("--save-log", type=Path, default=None, help="Also save a NavLog .npz here (input for animate.py)")
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


def _initial_pose_fields(initial_pose: list[float] | None) -> dict:
    if initial_pose is None:
        return {}
    x, y, yaw_deg = initial_pose
    return {"initial_position": (x, y), "initial_yaw": float(np.deg2rad(yaw_deg))}


# Without an absolute sensor the navigation frame is defined by the start pose, so the
# EKF's (0, 0, 0) start is exact in its own frame. A small prior also keeps uncertainty
# out of the unobservable global position/heading (see doc/visual_odometry.md §7).
ANCHORED_INITIAL_POSITION_STD = 0.01  # [m]
ANCHORED_INITIAL_YAW_STD_DEG = 0.1


def _ekf_initial_pose_std(args: argparse.Namespace) -> dict:
    position_std = args.ekf_initial_position_std
    yaw_std_deg = args.ekf_initial_yaw_std_deg
    if args.no_gnss_update:
        position_std = ANCHORED_INITIAL_POSITION_STD if position_std is None else position_std
        yaw_std_deg = ANCHORED_INITIAL_YAW_STD_DEG if yaw_std_deg is None else yaw_std_deg
    fields = {}
    if position_std is not None:
        fields["initial_position_std"] = position_std
    if yaw_std_deg is not None:
        fields["initial_yaw_std_rad"] = float(np.deg2rad(yaw_std_deg))
    return fields


def main() -> None:
    args = build_argument_parser().parse_args()
    simulator_config = SimulatorConfig(
        total_time=args.total_time,
        dt=args.dt,
        gnss_period=args.gnss_period,
        gnss_dropout_probability=args.gnss_dropout,
        seed=args.seed,
        scenario=args.scenario,
        forward_back_distance=args.forward_back_distance,
        accel_noise_density=args.accel_noise_density,
        gyro_noise_density=args.gyro_noise_density,
        initial_accel_bias=tuple(args.initial_accel_bias),
        initial_gyro_bias=float(np.deg2rad(args.initial_gyro_bias_dps)),
        vo_period=args.vo_period,
        vo_translation_std=args.vo_translation_std,
        vo_translation_std_per_m=args.vo_translation_std_per_m,
        vo_yaw_std=float(np.deg2rad(args.vo_yaw_std_deg)),
        vo_yaw_std_per_rad=args.vo_yaw_std_per_rad,
        vo_dropout_probability=args.vo_dropout,
        **_initial_pose_fields(args.initial_pose),
    )
    scenario = simulate_scenario(simulator_config)
    ekf_config = EkfConfig(
        accel_noise_std=simulator_config.accel_noise_std,
        gyro_noise_std=simulator_config.gyro_noise_std,
        accel_bias_walk_std=simulator_config.accel_bias_walk_std,
        gyro_bias_walk_std=simulator_config.gyro_bias_walk_std,
        gnss_position_std=simulator_config.gnss_position_std,
        gnss_velocity_std=simulator_config.gnss_velocity_std,
        **_ekf_initial_pose_std(args),
    )
    conditions_path = save_run_conditions(args.output_dir / "run_conditions.toml", simulator_config, ekf_config)
    print(f"saved run conditions: {conditions_path}")

    vo_inputs = {}
    if args.use_vo:
        vo_inputs = {
            "vo_from_index": np.asarray(scenario["vo_from_index"]),
            "vo_to_index": np.asarray(scenario["vo_to_index"]),
            "vo_measurements": np.asarray(scenario["vo_delta_measurements"]),
            "vo_covariances": np.asarray(scenario["vo_covariances"]),
        }
        print(f"fusing {len(vo_inputs['vo_measurements'])} VO delta poses")
    gnss_used = np.asarray(scenario["gnss_available"]).copy()
    if args.no_gnss_update:
        gnss_used[:] = False
        print("GNSS is simulated but not fused; the EKF navigates in its start-anchored frame")
    result = run_filter(
        imu_measurements=np.asarray(scenario["imu_measurements"]),
        gnss_measurements=np.asarray(scenario["gnss_measurements"]),
        gnss_available=gnss_used,
        dt=float(scenario["dt"]),
        config=ekf_config,
        **vo_inputs,
    )

    if args.save_log is not None:
        nav_log = NavLog.from_run(
            time=np.asarray(scenario["time"]),
            result=result,
            gnss_measurements=np.asarray(scenario["gnss_measurements"]),
            gnss_available=gnss_used,
            config=ekf_config,
            truth_states=np.asarray(scenario["truth_states"]),
            imu_measurements=np.asarray(scenario["imu_measurements"]),
            metadata={"source": "simulation", "simulator_config": dataclasses.asdict(simulator_config), "vo_fused": args.use_vo,
                      "gnss_fused": not args.no_gnss_update},
        )
        print(f"saved navigation log: {nav_log.save(args.save_log)}")

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

    bias_figure = create_bias_timeseries_figure(
        time=np.asarray(scenario["time"]),
        state_estimates=np.asarray(result["state_estimates"]),
        covariances=np.asarray(result["covariances"]),
        truth_states=np.asarray(scenario["truth_states"]),
    )
    bias_output_path = save_figure(bias_figure, args.output_dir / "ekf_bias_timeseries.png")
    print(f"saved figure: {bias_output_path}")

    if args.no_gnss_update:
        truth_states = np.asarray(scenario["truth_states"])
        estimates = np.asarray(result["state_estimates"])
        alignment = align_trajectory(estimates[:, [IDX_X, IDX_Y]], estimates[:, IDX_YAW],
                                     truth_states[:, [IDX_X, IDX_Y]], truth_states[:, IDX_YAW])
        alignment_figure = create_frame_alignment_figure(
            time=np.asarray(scenario["time"]),
            truth_states=truth_states,
            state_estimates=estimates,
            alignment=alignment,
            gnss_measurements=np.asarray(scenario["gnss_measurements"]),
            gnss_available=np.asarray(scenario["gnss_available"]),
        )
        print(f"saved figure: {save_figure(alignment_figure, args.output_dir / 'ekf_frame_alignment.png')}")
        print(f"true start pose (nav -> world): rotation {np.rad2deg(alignment.anchor.yaw):.3f} deg, "
              f"translation ({alignment.anchor.translation[0]:.3f}, {alignment.anchor.translation[1]:.3f}) m")
        print(f"best-fit rigid transform     : rotation {np.rad2deg(alignment.fitted.yaw):.3f} deg, "
              f"translation ({alignment.fitted.translation[0]:.3f}, {alignment.fitted.translation[1]:.3f}) m")
        print(f"ate_best_fit_rmse_m: {alignment.ate_fitted_rmse:.6f}")
        print(f"ate_true_start_rmse_m: {alignment.ate_anchored_rmse:.6f}")
        print(f"yaw_true_start_rmse_deg: {np.rad2deg(alignment.yaw_anchored_rmse):.6f}")
        print("errors below compare the nav-frame estimate directly with world-frame truth:")

    for key, value in summarize_errors(np.asarray(scenario["truth_states"]), np.asarray(result["state_estimates"])).items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
