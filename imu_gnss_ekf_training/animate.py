"""CLI: render a saved navigation log (`NavLog` .npz) as a 2D movie.

Usage
-----
    uv run python -m imu_gnss_ekf_training.demo --save-log outputs/run1/nav_log.npz
    uv run python -m imu_gnss_ekf_training.animate outputs/run1/nav_log.npz --output outputs/run1/nav.mp4 --speed 2
    uv run python -m imu_gnss_ekf_training.animate outputs/run1/nav_log.npz --output outputs/run1/nav.gif --follow 20 --trail-seconds 15
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt

from .animation import NavAnimator
from .navlog import NavLog

logger = logging.getLogger(__name__)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render an EKF navigation log as a 2D movie with covariance ellipses.")
    parser.add_argument("log", type=Path, help="NavLog .npz written by demo.py / run_ekf_from_csv.py --save-log")
    parser.add_argument("--output", type=Path, default=None, help="Output .mp4 (ffmpeg) or .gif; default <log dir>/nav_animation.mp4")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed relative to real time")
    parser.add_argument("--confidence", type=float, default=0.95, help="Probability mass inside drawn ellipses")
    parser.add_argument("--trail-seconds", type=float, default=None, help="Only draw the last N seconds of trajectory (default: all)")
    parser.add_argument("--follow", type=float, default=None, metavar="HALF_WIDTH_M", help="Camera follows the estimate, half-width [m]")
    parser.add_argument("--gnss-hold-seconds", type=float, default=1.0, help="How long the latest GNSS fix and its prior ellipse stay drawn")
    parser.add_argument("--no-prior", action="store_true", help="Do not draw the pre-update (prior) ellipse")
    parser.add_argument("--start", type=float, default=None, help="Start time [s]")
    parser.add_argument("--end", type=float, default=None, help="End time [s]")
    parser.add_argument("--dpi", type=int, default=100)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_argument_parser().parse_args()
    plt.switch_backend("Agg")

    log = NavLog.load(args.log)
    output = args.output if args.output is not None else args.log.parent / "nav_animation.mp4"
    animator = NavAnimator(
        log,
        fps=args.fps,
        speed=args.speed,
        confidence=args.confidence,
        trail_seconds=args.trail_seconds,
        follow_half_width=args.follow,
        show_prior=not args.no_prior,
        gnss_hold_seconds=args.gnss_hold_seconds,
        start_time=args.start,
        end_time=args.end,
    )
    num_frames = len(animator.frame_indices)
    logger.info("loaded %s: %d steps, %d GNSS fixes; rendering %d frames", args.log, log.num_steps, int(log.gnss_available.sum()), num_frames)

    step = max(1, num_frames // 10)

    def report(frame: int, total: int) -> None:
        if frame % step == 0 or frame == total - 1:
            logger.info("frame %d / %d", frame + 1, total)

    path = animator.save(output, dpi=args.dpi, progress_callback=report)
    logger.info("saved movie: %s", path)


if __name__ == "__main__":
    main()
