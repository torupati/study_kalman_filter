"""IMU + visual-odometry navigation without GNSS: error growth and VO noise sweep.

Supports doc/vo_only_navigation.md. GNSS is simulated but not fused; the EKF starts at
(0, 0, 0) with a tight prior, i.e. it navigates in a frame anchored at the true start
pose, which is (30, -20) m heading 40 deg in the world. Estimates are compared with
truth after mapping them with the true start pose (`alignment.align_trajectory`).

1. The reference run, identical to
       uv run python -m imu_gnss_ekf_training.demo --scenario demo2 --dt 0.01 \\
         --use-vo --no-gnss-update --initial-pose 30 -20 40
   -> figures/vo_only_demo2_alignment.png, figures/vo_only_demo2_error_growth.png
2. A Monte Carlo sweep over VO noise (all four `vo_*` std parameters scaled by 1, 3, 10)
   on demo2 and circle, with VO-only dead reckoning as a baseline
   -> figures/vo_only_noise_sweep.png and a markdown table in the log.
3. Why the IMU adds little to VO here: demo2 with VO noise x10, with the initial gyro
   bias unknown (default) or known to the EKF, for two gyro bias random-walk levels.

Usage:
    uv run python -m imu_gnss_ekf_training.misc.analyze_vo_only
    uv run python -m imu_gnss_ekf_training.misc.analyze_vo_only --seeds 50
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from imu_gnss_ekf_training.alignment import align_trajectory
from imu_gnss_ekf_training.ekf import IDX_BG, IDX_VX, IDX_VY, IDX_X, IDX_Y, IDX_YAW, STATE_SIZE, EkfConfig, run_filter, wrap_angle
from imu_gnss_ekf_training.plotting import create_frame_alignment_figure
from imu_gnss_ekf_training.simulator import SimulatorConfig, compose_pose_2d, simulate_scenario

logger = logging.getLogger(__name__)

FIGURE_DIR = Path(__file__).resolve().parent.parent / "figures"

DT = 0.01  # 100 Hz IMU
TOTAL_TIME = 60.0
START_POSITION = (30.0, -20.0)
START_YAW_DEG = 40.0
# same anchoring as demo.py --no-gnss-update
INITIAL_POSITION_STD = 0.01  # [m]
INITIAL_YAW_STD_DEG = 0.1
NOISE_SCALES = (1.0, 3.0, 10.0)
SCENARIOS = ("demo2", "circle")
NOISE_COLORS = {1.0: "tab:blue", 3.0: "tab:orange", 10.0: "tab:red"}
STOPPED_SPEED = 0.05  # [m/s] below this the vehicle counts as stopped
KNOWN_GYRO_BIAS_STD_DPS = 0.01
GYRO_BIAS_WALKS = (SimulatorConfig().gyro_bias_walk_std, 1e-4)  # [rad/s/sqrt(s)] default and a steady gyro


def simulator_config(scenario: str, seed: int, noise_scale: float = 1.0) -> SimulatorConfig:
    base = SimulatorConfig()
    return SimulatorConfig(
        scenario=scenario,
        dt=DT,
        total_time=TOTAL_TIME,
        seed=seed,
        initial_position=START_POSITION,
        initial_yaw=float(np.deg2rad(START_YAW_DEG)),
        vo_translation_std=base.vo_translation_std * noise_scale,
        vo_translation_std_per_m=base.vo_translation_std_per_m * noise_scale,
        vo_yaw_std=base.vo_yaw_std * noise_scale,
        vo_yaw_std_per_rad=base.vo_yaw_std_per_rad * noise_scale,
    )


def run_vo_only(config: SimulatorConfig, known_gyro_bias: bool = False) -> dict:
    """Simulate, run the EKF with IMU + VO only, and compare with truth via the true start pose.

    With `known_gyro_bias` the EKF starts at the true initial gyro bias with a tight prior.
    """
    data = simulate_scenario(config)
    initial_state = np.zeros(STATE_SIZE)
    gyro_bias_prior = {}
    if known_gyro_bias:
        initial_state[IDX_BG] = config.initial_gyro_bias
        gyro_bias_prior = {"initial_gyro_bias_std": float(np.deg2rad(KNOWN_GYRO_BIAS_STD_DPS))}
    ekf_config = EkfConfig(
        accel_noise_std=config.accel_noise_std,
        gyro_noise_std=config.gyro_noise_std,
        accel_bias_walk_std=config.accel_bias_walk_std,
        gyro_bias_walk_std=config.gyro_bias_walk_std,
        gnss_position_std=config.gnss_position_std,
        gnss_velocity_std=config.gnss_velocity_std,
        initial_position_std=INITIAL_POSITION_STD,
        initial_yaw_std_rad=float(np.deg2rad(INITIAL_YAW_STD_DEG)),
        **gyro_bias_prior,
    )
    result = run_filter(
        imu_measurements=data["imu_measurements"],
        gnss_measurements=data["gnss_measurements"],
        gnss_available=np.zeros_like(data["gnss_available"]),
        dt=data["dt"],
        config=ekf_config,
        initial_state=initial_state,
        vo_from_index=data["vo_from_index"],
        vo_to_index=data["vo_to_index"],
        vo_measurements=data["vo_delta_measurements"],
        vo_covariances=data["vo_covariances"],
    )
    truth = data["truth_states"]
    estimates = result["state_estimates"]
    alignment = align_trajectory(estimates[:, [IDX_X, IDX_Y]], estimates[:, IDX_YAW], truth[:, [IDX_X, IDX_Y]], truth[:, IDX_YAW])
    covariances = result["covariances"]
    return {
        "data": data,
        "result": result,
        "alignment": alignment,
        "position_error": np.linalg.norm(alignment.anchored_xy - truth[:, [IDX_X, IDX_Y]], axis=1),
        "yaw_error": wrap_angle(alignment.anchored_yaw - truth[:, IDX_YAW]),
        # the trace and the yaw variance do not change under the nav -> world rotation
        "position_sigma": np.sqrt(covariances[:, IDX_X, IDX_X] + covariances[:, IDX_Y, IDX_Y]),
        "yaw_sigma": np.sqrt(covariances[:, IDX_YAW, IDX_YAW]),
    }


def vo_dead_reckoning_error(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Chain the noisy VO deltas from the true start pose; position/yaw error at each VO frame."""
    truth = data["truth_states"][:, [IDX_X, IDX_Y, IDX_YAW]]
    pose = truth[data["vo_from_index"][0]]
    poses = [pose]
    for delta in data["vo_delta_measurements"]:
        pose = compose_pose_2d(pose, delta)
        poses.append(pose)
    poses = np.array(poses)
    reference = truth[np.r_[data["vo_from_index"][0], data["vo_to_index"]]]
    return np.linalg.norm(poses[:, :2] - reference[:, :2], axis=1), wrap_angle(poses[:, 2] - reference[:, 2])


def figure_error_growth(run: dict):
    """Reference run: speed, position error and yaw error with their 1-sigma, stopped period shaded."""
    data = run["data"]
    time = data["time"]
    truth = data["truth_states"]
    speed = np.hypot(truth[:, IDX_VX], truth[:, IDX_VY])
    stopped = speed < STOPPED_SPEED

    figure, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
    figure.suptitle("IMU + VO without GNSS (demo2): error after mapping with the true start pose")
    panels = (
        (axes[0], speed, None, "speed [m/s]", "true speed"),
        (axes[1], run["position_error"], run["position_sigma"], "position error [m]", "|error|"),
        (axes[2], np.rad2deg(run["yaw_error"]), np.rad2deg(run["yaw_sigma"]), "yaw error [deg]", "error"),
    )
    for ax, value, sigma, ylabel, label in panels:
        ax.fill_between(time, 0, 1, where=stopped, transform=ax.get_xaxis_transform(), color="tab:gray", alpha=0.15,
                        label="stopped")
        ax.plot(time, value, color="tab:blue", label=label)
        if sigma is not None:
            lower = 0.0 * sigma if ylabel.startswith("position") else -sigma
            ax.fill_between(time, lower, sigma, color="tab:blue", alpha=0.15, label="EKF 1σ")
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper left")
        ax.grid(True)
    axes[-1].set_xlabel("time [s]")
    return figure


def monte_carlo(num_seeds: int) -> dict:
    """RMS over seeds of the error and of the EKF sigma, per (scenario, noise scale)."""
    results = {}
    for scenario in SCENARIOS:
        for scale in NOISE_SCALES:
            runs = [run_vo_only(simulator_config(scenario, seed, scale)) for seed in range(num_seeds)]
            dead_reckoning = [vo_dead_reckoning_error(run["data"]) for run in runs]

            def rms(values):
                return np.sqrt(np.mean(np.square(values), axis=0))

            results[scenario, scale] = {
                "time": runs[0]["data"]["time"],
                "position_error": rms([run["position_error"] for run in runs]),
                "position_sigma": rms([run["position_sigma"] for run in runs]),
                "yaw_error": rms([run["yaw_error"] for run in runs]),
                "yaw_sigma": rms([run["yaw_sigma"] for run in runs]),
                "dr_position_error": rms([position for position, _ in dead_reckoning])[-1],
                "dr_yaw_error": rms([yaw for _, yaw in dead_reckoning])[-1],
                "fit_rotation_error": rms([wrap_angle(run["alignment"].fitted.yaw - run["alignment"].anchor.yaw) for run in runs]),
            }
            logger.info("monte carlo %s x%g done", scenario, scale)
    return results


def figure_noise_sweep(results: dict):
    figure, axes = plt.subplots(2, len(SCENARIOS), figsize=(14, 9), sharex=True, constrained_layout=True)
    figure.suptitle("IMU + VO without GNSS: RMS over seeds (solid: actual error, dashed: EKF 1σ)")
    for column, scenario in enumerate(SCENARIOS):
        for scale in NOISE_SCALES:
            entry = results[scenario, scale]
            color = NOISE_COLORS[scale]
            axes[0, column].plot(entry["time"], entry["position_error"], color=color, label=f"VO noise ×{scale:g}")
            axes[0, column].plot(entry["time"], entry["position_sigma"], color=color, linestyle="--")
            axes[1, column].plot(entry["time"], np.rad2deg(entry["yaw_error"]), color=color, label=f"VO noise ×{scale:g}")
            axes[1, column].plot(entry["time"], np.rad2deg(entry["yaw_sigma"]), color=color, linestyle="--")
        axes[0, column].set_title(scenario)
        axes[0, column].set_ylabel("position error [m]")
        axes[1, column].set_ylabel("yaw error [deg]")
        axes[1, column].set_xlabel("time [s]")
        for ax in axes[:, column]:
            ax.set_yscale("log")
            ax.set_ylim(bottom=1e-2)
            ax.legend(loc="lower right")
            ax.grid(True)
    return figure


def sweep_table(results: dict, num_seeds: int) -> str:
    lines = [
        f"RMS over {num_seeds} seeds at t = {TOTAL_TIME:g} s. Position/yaw: EKF error / EKF 1σ; "
        "VO DR: chained VO deltas only; fit: best-fit rotation minus true start heading.",
        "",
        "| scenario | VO noise | position [m] | VO DR position [m] | yaw [deg] | VO DR yaw [deg] | fit rotation error [deg] |",
        "|---|---|---|---|---|---|---|",
    ]
    for (scenario, scale), entry in results.items():
        lines.append(
            f"| {scenario} | ×{scale:g} | {entry['position_error'][-1]:.2f} / {entry['position_sigma'][-1]:.2f} "
            f"| {entry['dr_position_error']:.2f} "
            f"| {np.rad2deg(entry['yaw_error'][-1]):.2f} / {np.rad2deg(entry['yaw_sigma'][-1]):.2f} "
            f"| {np.rad2deg(entry['dr_yaw_error']):.2f} | {np.rad2deg(entry['fit_rotation_error']):.2f} |"
        )
    return "\n".join(lines)


def gyro_bias_table(num_seeds: int, noise_scale: float = 10.0) -> str:
    lines = [
        f"demo2, VO noise ×{noise_scale:g}, RMS over {num_seeds} seeds at t = {TOTAL_TIME:g} s (EKF error / EKF 1σ).",
        "",
        "| gyro bias walk [deg/s/√s] | initial gyro bias in EKF | position [m] | yaw [deg] |",
        "|---|---|---|---|",
    ]
    for walk in GYRO_BIAS_WALKS:
        for known in (False, True):
            runs = [
                run_vo_only(dataclasses.replace(simulator_config("demo2", seed, noise_scale), gyro_bias_walk_std=walk), known_gyro_bias=known)
                for seed in range(num_seeds)
            ]

            def rms(key):
                return np.sqrt(np.mean([run[key][-1] ** 2 for run in runs]))

            prior = f"known (1σ {KNOWN_GYRO_BIAS_STD_DPS:g} deg/s)" if known else "unknown (0, 1σ 2.9 deg/s)"
            lines.append(
                f"| {np.rad2deg(walk):.4f} | {prior} | {rms('position_error'):.2f} / {rms('position_sigma'):.2f} "
                f"| {np.rad2deg(rms('yaw_error')):.2f} / {np.rad2deg(rms('yaw_sigma')):.2f} |"
            )
    return "\n".join(lines)


def growth_table(run: dict, times: tuple[float, ...] = (0, 8, 16, 24, 32, 36, 48, 60)) -> str:
    data = run["data"]
    truth = data["truth_states"]
    speed = np.hypot(truth[:, IDX_VX], truth[:, IDX_VY])
    travelled = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(truth[:, [IDX_X, IDX_Y]], axis=0), axis=1))]
    from_start = np.linalg.norm(truth[:, [IDX_X, IDX_Y]] - truth[0, [IDX_X, IDX_Y]], axis=1)
    lines = [
        "| t [s] | speed [m/s] | travelled [m] | from start [m] | position error / 1σ [m] | yaw error / 1σ [deg] |",
        "|---|---|---|---|---|---|",
    ]
    for t in times:
        k = int(round(t / DT))
        lines.append(
            f"| {t:g} | {speed[k]:.2f} | {travelled[k]:.1f} | {from_start[k]:.1f} "
            f"| {run['position_error'][k]:.2f} / {run['position_sigma'][k]:.2f} "
            f"| {np.rad2deg(run['yaw_error'][k]):+.2f} / {np.rad2deg(run['yaw_sigma'][k]):.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=7, help="Seed of the reference run")
    parser.add_argument("--seeds", type=int, default=20, help="Number of Monte Carlo seeds for the noise sweep")
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    args = parser.parse_args()
    plt.switch_backend("Agg")
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    def save(figure, name: str) -> None:
        path = args.figure_dir / name
        figure.savefig(path, dpi=100)
        plt.close(figure)
        logger.info("saved %s", path)

    config = simulator_config("demo2", args.seed)
    run = run_vo_only(config)
    alignment = run["alignment"]
    logger.info("reference run config: %s", dataclasses.asdict(config))
    logger.info(
        "reference run: true start rotation %.3f deg, translation (%.3f, %.3f) m; best fit %.3f deg, (%.3f, %.3f) m; "
        "ATE best fit %.4f m, ATE true start %.4f m, yaw RMSE %.3f deg",
        np.rad2deg(alignment.anchor.yaw), *alignment.anchor.translation,
        np.rad2deg(alignment.fitted.yaw), *alignment.fitted.translation,
        alignment.ate_fitted_rmse, alignment.ate_anchored_rmse, np.rad2deg(alignment.yaw_anchored_rmse),
    )
    data = run["data"]
    save(
        create_frame_alignment_figure(
            time=data["time"], truth_states=data["truth_states"], state_estimates=run["result"]["state_estimates"],
            alignment=alignment, gnss_measurements=data["gnss_measurements"], gnss_available=data["gnss_available"],
        ),
        "vo_only_demo2_alignment.png",
    )
    save(figure_error_growth(run), "vo_only_demo2_error_growth.png")
    logger.info("reference run error growth:\n%s", growth_table(run))

    results = monte_carlo(args.seeds)
    save(figure_noise_sweep(results), "vo_only_noise_sweep.png")
    logger.info("noise sweep:\n%s", sweep_table(results, args.seeds))
    logger.info("gyro bias:\n%s", gyro_bias_table(args.seeds))


if __name__ == "__main__":
    main()
