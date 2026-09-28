"""Bias-estimation convergence per scenario (stationary, forward_back, circle).

Supports doc/bias_convergence.md. Each scenario is simulated with a known, constant
true bias (bias random walk off in the simulator; the EKF keeps its default walk Q),
100 Hz IMU, and GNSS every 0.5 s without dropout. For every run the script reports
the bias error against its +/-2 sigma band, the Kalman gain that feeds each bias
state, and yaw/bias correlations; a Monte Carlo over seeds checks consistency.

The Kalman gain is not logged by `run_filter`; it is recovered from the NavLog as
K = P_prior H^T S^-1 (H selects [x, y, vx, vy], so P_prior H^T = P_prior[:, :4]).

As a reference, the same covariance/gain recursion is also evaluated along the TRUE
trajectory with noise-free IMU input ("truth-linearized"). That is the textbook
covariance (observability) analysis: its gains show which states the measurements can
inform in principle, without the EKF's linearization errors.

Usage:
    uv run python -m imu_gnss_ekf_training.misc.analyze_bias_convergence
    uv run python -m imu_gnss_ekf_training.misc.analyze_bias_convergence --seeds 50 --movies
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from imu_gnss_ekf_training.animation import BiasAnimator
from imu_gnss_ekf_training.ekf import IDX_BAX, IDX_BAY, IDX_BG, IDX_YAW, MEAS_SIZE, EkfConfig, ImuGnssEkf, run_filter
from imu_gnss_ekf_training.navlog import NavLog
from imu_gnss_ekf_training.plotting import GRAVITY_MPS2
from imu_gnss_ekf_training.simulator import SimulatorConfig, simulate_scenario

logger = logging.getLogger(__name__)

FIGURE_DIR = Path(__file__).resolve().parent.parent / "figures"

# (scenario, total_time [s])
CASES = (("stationary", 30.0), ("forward_back", 30.0), ("circle", 40.0))

TRUE_ACCEL_BIAS = (0.2, -0.15)  # m/s^2, about (20, -15) mG
TRUE_GYRO_BIAS_DPS = 1.0

MG = 1000.0 / GRAVITY_MPS2
DPS = float(np.rad2deg(1.0))
# (state index, label, unit, factor from SI)
BIASES = ((IDX_BG, "gyro z bias", "deg/s", DPS), (IDX_BAX, "accel x bias", "mG", MG), (IDX_BAY, "accel y bias", "mG", MG))
COLORS = {"stationary": "tab:gray", "forward_back": "tab:green", "circle": "tab:purple"}


DT = 0.01  # 100 Hz IMU


def simulate_and_filter(scenario: str, total_time: float, seed: int) -> tuple[NavLog, dict]:
    """Simulate and run the EKF; returns the log and the raw simulator output."""
    sim = SimulatorConfig(
        scenario=scenario,
        total_time=total_time,
        dt=DT,
        gnss_period=0.5,
        gnss_dropout_probability=0.0,
        seed=seed,
        initial_accel_bias=TRUE_ACCEL_BIAS,
        initial_gyro_bias=float(np.deg2rad(TRUE_GYRO_BIAS_DPS)),
        accel_bias_walk_std=0.0,
        gyro_bias_walk_std=0.0,
    )
    data = simulate_scenario(sim)
    config = EkfConfig()
    result = run_filter(data["imu_measurements"], data["gnss_measurements"], data["gnss_available"], DT, config)
    log = NavLog.from_run(
        time=data["time"],
        result=result,
        gnss_measurements=data["gnss_measurements"],
        gnss_available=data["gnss_available"],
        config=config,
        truth_states=data["truth_states"],
        imu_measurements=data["imu_measurements"],
        metadata={"source": "analyze_bias_convergence", "scenario": scenario, "seed": seed},
    )
    return log, data


def truth_linearized(data: dict, config: EkfConfig, dt: float) -> dict[str, np.ndarray]:
    """Covariance and gain recursion of the EKF, linearized about the true state with noise-free IMU input.

    `ImuGnssEkf.predict` builds F from (IMU - bias estimate); passing the true state and
    the noise-free IMU sample (true motion + true bias) makes that exactly the true
    body-frame acceleration. The covariance update does not depend on the state.
    """
    ekf = ImuGnssEkf(config)
    truth = data["truth_states"]
    noise_free_imu = np.column_stack((data["true_accel_body"] + truth[:, [IDX_BAX, IDX_BAY]], data["true_yaw_rate"] + truth[:, IDX_BG]))
    num_steps, state_size = truth.shape
    covariance = np.diag([
        config.initial_position_std**2, config.initial_position_std**2,
        config.initial_velocity_std**2, config.initial_velocity_std**2,
        config.initial_yaw_std_rad**2,
        config.initial_accel_bias_std**2, config.initial_accel_bias_std**2,
        config.initial_gyro_bias_std**2,
    ])
    covariances = np.zeros((num_steps, state_size, state_size))
    gains = np.full((num_steps, state_size, MEAS_SIZE), np.nan)
    for k in range(num_steps):
        if k > 0:
            _, covariance = ekf.predict(truth[k - 1], covariance, noise_free_imu[k - 1], dt)
        if data["gnss_available"][k]:
            innovation_covariance = ekf.H @ covariance @ ekf.H.T + ekf.R
            gains[k] = np.linalg.solve(innovation_covariance.T, (covariance @ ekf.H.T).T).T
            _, covariance = ekf.update(truth[k], covariance, ekf.H @ truth[k])
        covariances[k] = covariance
    return {"covariances": covariances, "gains": gains}


def estimation_error(log: NavLog) -> np.ndarray:
    error = log.state_estimates - log.truth_states
    error[:, IDX_YAW] = np.arctan2(np.sin(error[:, IDX_YAW]), np.cos(error[:, IDX_YAW]))
    return error


def kalman_gains(log: NavLog) -> np.ndarray:
    """(N, 8, 4) gain applied at each GNSS update; NaN on steps without one."""
    gains = np.full((log.num_steps, log.state_estimates.shape[1], MEAS_SIZE), np.nan)
    updates = np.flatnonzero(log.gnss_available)
    gains[updates] = np.linalg.solve(
        np.swapaxes(log.innovation_covariances[updates], 1, 2), np.swapaxes(log.prior_covariances[updates][:, :, :MEAS_SIZE], 1, 2)
    ).swapaxes(1, 2)
    return gains


def correlation(log: NavLog, i: int, j: int) -> np.ndarray:
    P = log.covariances
    return P[:, i, j] / np.sqrt(P[:, i, i] * P[:, j, j])


def figure_errors(logs: dict[str, NavLog], ideals: dict[str, dict]) -> plt.Figure:
    figure, axes = plt.subplots(len(BIASES), len(logs), figsize=(13, 8), sharex="col", constrained_layout=True)
    for row, (index, label, unit, factor) in enumerate(BIASES):
        # Same y range across scenarios: the largest error after the first second, or 3x the true bias.
        true_bias = factor * (np.deg2rad(TRUE_GYRO_BIAS_DPS) if index == IDX_BG else max(abs(b) for b in TRUE_ACCEL_BIAS))
        largest = max(float(np.max(np.abs(factor * estimation_error(log)[log.time > 1.0, index]))) for log in logs.values())
        limit = max(3.0 * true_bias, 1.15 * largest)
        for col, (scenario, log) in enumerate(logs.items()):
            ax = axes[row, col]
            error = estimation_error(log)
            band = 2.0 * factor * np.sqrt(log.covariances[:, index, index])
            ideal = 2.0 * factor * np.sqrt(ideals[scenario]["covariances"][:, index, index])
            ax.fill_between(log.time, -band, band, color="tab:blue", alpha=0.2, linewidth=0, label="±2σ EKF")
            ax.plot(log.time, ideal, color="black", linestyle="--", linewidth=0.8, label="±2σ truth-linearized")
            ax.plot(log.time, -ideal, color="black", linestyle="--", linewidth=0.8)
            ax.plot(log.time, factor * error[:, index], color="tab:blue", linewidth=1.2, label="estimate − truth")
            ax.axhline(0.0, color="black", linewidth=0.6)
            ax.set_ylim(-limit, limit)
            ax.grid(True)
            if row == 0:
                ax.set_title(scenario)
            if col == 0:
                ax.set_ylabel(f"{label} error [{unit}]")
            if row == len(BIASES) - 1:
                ax.set_xlabel("time [s]")
    axes[0, 0].legend(loc="lower right", fontsize=8)
    return figure


def figure_gains(logs: dict[str, NavLog], ideals: dict[str, dict]) -> plt.Figure:
    """|K| rows for yaw and each bias: EKF (solid) vs. truth-linearized (dashed; exact zeros are not drawn on the log axis)."""
    rows = ((IDX_YAW, "yaw", "deg", DPS),) + BIASES
    figure, axes = plt.subplots(len(rows), 1, figsize=(10, 10), sharex=True, constrained_layout=True)
    always_zero: dict[int, list[str]] = {index: [] for index, *_ in rows}
    for scenario, log in logs.items():
        gains = kalman_gains(log)
        ideal_gains = ideals[scenario]["gains"]
        updates = log.gnss_available
        for ax, (index, label, unit, factor) in zip(axes, rows):
            # Scaled to display units per innovation unit (m or m/s): how far one metre / one m/s of
            # GNSS innovation moves the estimate. Exact zeros cannot be drawn on a log axis.
            gain = factor * np.linalg.norm(gains[updates, index, :], axis=1)
            ideal = factor * np.linalg.norm(ideal_gains[updates, index, :], axis=1)
            ax.semilogy(log.time[updates], np.where(gain > 0, gain, np.nan),
                        color=COLORS[scenario], linewidth=1.2, label=f"{scenario} EKF")
            ax.semilogy(log.time[updates], np.where(ideal > 0, ideal, np.nan),
                        color=COLORS[scenario], linewidth=1.0, linestyle="--", label=f"{scenario} truth-linearized")
            if np.all(ideal == 0.0):
                always_zero[index].append(scenario)
    for ax, (index, *_) in zip(axes, rows):
        if always_zero[index]:
            ax.text(0.01, 0.04, f"truth-linearized gain is exactly 0 for: {', '.join(always_zero[index])}",
                    transform=ax.transAxes, fontsize=9, bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    for ax, (_, label, unit, _) in zip(axes, rows):
        ax.set_ylabel(f"|K| {label}\n[{unit} per m, m/s]")
        ax.grid(True, which="both", linewidth=0.4)
        ax.set_ylim(bottom=1e-3)
    axes[0].legend(fontsize=7, ncol=3)
    axes[-1].set_xlabel("time [s]")
    return figure


def figure_yaw_coupling(logs: dict[str, NavLog], ideals: dict[str, dict]) -> plt.Figure:
    figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    for scenario, log in logs.items():
        color = COLORS[scenario]
        axes[0].plot(log.time, DPS * np.sqrt(log.covariances[:, IDX_YAW, IDX_YAW]), color=color, label=f"{scenario} EKF")
        axes[0].plot(log.time, DPS * np.sqrt(ideals[scenario]["covariances"][:, IDX_YAW, IDX_YAW]), color=color, linestyle="--",
                     linewidth=1.0, label=f"{scenario} truth-linearized")
        axes[1].plot(log.time, correlation(log, IDX_YAW, IDX_BAX), color=color, label=scenario)
        axes[2].plot(log.time, correlation(log, IDX_YAW, IDX_BG), color=color, label=scenario)
    axes[0].set_ylabel("yaw σ [deg]")
    axes[1].set_ylabel("corr(yaw, accel x bias)")
    axes[2].set_ylabel("corr(yaw, gyro z bias)")
    for ax in axes[1:]:
        ax.set_ylim(-1.05, 1.05)
    for ax in axes:
        ax.grid(True)
    axes[0].legend(fontsize=7, ncol=3)
    axes[-1].set_xlabel("time [s]")
    return figure


def monte_carlo_table(num_seeds: int) -> str:
    """Final-time RMS error vs. RMS filter sigma over seeds, as a Markdown table."""
    states = ((IDX_YAW, "yaw", "deg", DPS),) + BIASES
    lines = [
        "| scenario | " + " | ".join(f"{label} [{unit}] RMS err / σ" for _, label, unit, _ in states) + " |",
        "|---|" + "---|" * len(states),
    ]
    for scenario, total_time in CASES:
        errors, sigmas = [], []
        for seed in range(num_seeds):
            log, _ = simulate_and_filter(scenario, total_time, seed)
            errors.append(estimation_error(log)[-1])
            sigmas.append(np.sqrt(np.diag(log.covariances[-1])))
        errors, sigmas = np.array(errors), np.array(sigmas)
        cells = []
        for index, _, _, factor in states:
            rms_error = factor * np.sqrt(np.mean(errors[:, index] ** 2))
            rms_sigma = factor * np.sqrt(np.mean(sigmas[:, index] ** 2))
            cells.append(f"{rms_error:.2f} / {rms_sigma:.2f}")
        lines.append(f"| {scenario} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=7, help="Seed of the single runs shown in the figures")
    parser.add_argument("--seeds", type=int, default=30, help="Number of Monte Carlo seeds for the consistency table")
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/bias_convergence"), help="Where each scenario's nav_log.npz is saved")
    parser.add_argument("--movies", action="store_true", help="Also render <log-dir>/<scenario>/bias_animation.mp4")
    parser.add_argument("--fps", type=float, default=15.0)
    args = parser.parse_args()
    plt.switch_backend("Agg")

    runs = {scenario: simulate_and_filter(scenario, total_time, args.seed) for scenario, total_time in CASES}
    logs = {scenario: log for scenario, (log, _) in runs.items()}
    ideals = {scenario: truth_linearized(data, EkfConfig(), DT) for scenario, (_, data) in runs.items()}
    for scenario, log in logs.items():
        path = log.save(args.log_dir / scenario / "nav_log.npz")
        logger.info("saved %s", path)

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    for name, build in (("error", figure_errors), ("gain", figure_gains), ("yaw_coupling", figure_yaw_coupling)):
        figure = build(logs, ideals)
        path = args.figure_dir / f"bias_convergence_{name}.png"
        figure.savefig(path, dpi=100)
        plt.close(figure)
        logger.info("saved %s", path)

    for scenario, log in logs.items():
        error = estimation_error(log)
        sigma = np.sqrt(np.diagonal(log.covariances, axis1=1, axis2=2))
        summary = ", ".join(
            f"{label}: err {factor * error[-1, index]:+.2f} σ {factor * sigma[0, index]:.2f}→{factor * sigma[-1, index]:.2f} "
            f"(truth-linearized {factor * np.sqrt(ideals[scenario]['covariances'][-1, index, index]):.2f}) {unit}"
            for index, label, unit, factor in ((IDX_YAW, "yaw", "deg", DPS),) + BIASES
        )
        logger.info("%s (seed %d, final): %s; corr(yaw, bax) %+.2f", scenario, args.seed, summary, correlation(log, IDX_YAW, IDX_BAX)[-1])

    logger.info("Monte Carlo over %d seeds (final time):\n%s", args.seeds, monte_carlo_table(args.seeds))

    if args.movies:
        for scenario, log in logs.items():
            animator = BiasAnimator(log, fps=args.fps, speed=1.0)
            path = animator.save(args.log_dir / scenario / "bias_animation.mp4")
            plt.close(animator.figure)
            logger.info("saved %s", path)


if __name__ == "__main__":
    main()
