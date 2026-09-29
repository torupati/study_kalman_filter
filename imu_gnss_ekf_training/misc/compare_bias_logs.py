"""Overlay the bias estimates of several NavLog runs for comparison.

One axis per bias (accel x, accel y, gyro z); each log gets its own color, with the
estimate as a solid line, its 95% range (±1.96 std) as dotted lines, the truth (when
the log has it) as a dashed line, and optionally a ±1 std band. Labels default to the log's parent directory name, which
matches the `outputs/<set>/<scenario>/nav_log.npz` layout of misc/run_scenario.bash.

Usage:
    uv run python -m imu_gnss_ekf_training.misc.compare_bias_logs --log-dir outputs/scenarios2
    uv run python -m imu_gnss_ekf_training.misc.compare_bias_logs a/nav_log.npz b/nav_log.npz --labels A B --std
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from imu_gnss_ekf_training.ekf import IDX_BAX, IDX_BAY, IDX_BG
from imu_gnss_ekf_training.navlog import NavLog
from imu_gnss_ekf_training.plotting import GRAVITY_MPS2, save_figure

logger = logging.getLogger(__name__)

Z_95 = 1.959963984540054  # two-sided 95% quantile of the standard normal

# (state index, title, y label, factor from SI)
BIASES = (
    (IDX_BAX, "accel_x bias", "bias [G]", 1.0 / GRAVITY_MPS2),
    (IDX_BAY, "accel_y bias", "bias [G]", 1.0 / GRAVITY_MPS2),
    (IDX_BG, "gyro_z bias", "bias [deg/s]", float(np.rad2deg(1.0))),
)


def create_bias_comparison_figure(
    logs: list[NavLog], labels: list[str], show_std: bool = False, show_truth: bool = True, show_ci95: bool = True,
):
    figure, axes = plt.subplots(len(BIASES), 1, figsize=(11, 10), sharex=True, constrained_layout=True)
    figure.suptitle("EKF bias estimates (comparison)")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for log_index, (log, label) in enumerate(zip(logs, labels, strict=True)):
        color = colors[log_index % len(colors)]
        std = np.sqrt(np.diagonal(log.covariances, axis1=1, axis2=2))
        for ax, (index, _, _, scale) in zip(axes, BIASES, strict=True):
            estimate = log.state_estimates[:, index] * scale
            ax.plot(log.time, estimate, color=color, label=f"{label} ekf")
            band = std[:, index] * scale
            if show_std:
                ax.fill_between(log.time, estimate - band, estimate + band, color=color, alpha=0.15, label=f"{label} ±1 std")
            if show_ci95:
                ax.plot(log.time, estimate + Z_95 * band, color=color, linestyle=":", linewidth=1.0, label=f"{label} 95%")
                ax.plot(log.time, estimate - Z_95 * band, color=color, linestyle=":", linewidth=1.0)
            if show_truth and log.has_truth:
                ax.plot(log.time, log.truth_states[:, index] * scale, color=color, linestyle="--", label=f"{label} truth")

    for ax, (_, title, ylabel, _) in zip(axes, BIASES, strict=True):
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True)
    axes[0].legend(ncol=len(logs), fontsize="small")
    axes[-1].set_xlabel("time [s]")
    return figure


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Overlay bias estimates from multiple NavLog (.npz) files.")
    parser.add_argument("logs", type=Path, nargs="*", help="NavLog .npz files")
    parser.add_argument("--log-dir", type=Path, default=None, help="Also compare every <log-dir>/*/nav_log.npz")
    parser.add_argument("--labels", nargs="+", default=None, help="Legend labels (default: parent directory names)")
    parser.add_argument("--std", action="store_true", help="Draw ±1 std bands")
    parser.add_argument("--no-ci95", action="store_true", help="Do not draw the 95%% range (±1.96 std) lines")
    parser.add_argument("--no-truth", action="store_true", help="Do not draw truth even when available")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output image (default: <log-dir>/bias_comparison.png, or ./bias_comparison.png)",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = build_argument_parser()
    args = parser.parse_args()

    log_paths = list(args.logs)
    if args.log_dir is not None:
        log_paths += sorted(args.log_dir.glob("*/nav_log.npz"))
    if not log_paths:
        parser.error("no logs given (pass .npz files and/or --log-dir)")
    labels = args.labels if args.labels is not None else [path.parent.name for path in log_paths]
    if len(labels) != len(log_paths):
        parser.error(f"--labels has {len(labels)} entries but there are {len(log_paths)} logs")

    logs = []
    for path, label in zip(log_paths, labels, strict=True):
        logger.info("loading %s (%s)", path, label)
        logs.append(NavLog.load(path))

    figure = create_bias_comparison_figure(logs, labels, show_std=args.std, show_truth=not args.no_truth, show_ci95=not args.no_ci95)
    output = args.output
    if output is None:
        output = (args.log_dir if args.log_dir is not None else Path(".")) / "bias_comparison.png"
    logger.info("saved figure: %s", save_figure(figure, output))


if __name__ == "__main__":
    main()
