"""Animate a `NavLog`: the 2D trajectory with covariance ellipses, or the bias estimates.

`NavAnimator` (2D plane) and `BiasAnimator` (bias time series) precompute everything
per log step in `__init__`, create each matplotlib artist once, and `update(frame)`
only mutates those artists. Neither runs the filter.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.patches import Ellipse, Polygon

from .ekf import IDX_BAX, IDX_BAY, IDX_BG, IDX_X, IDX_Y, IDX_YAW
from .ellipse import covariance_ellipse, normal_1d_scale
from .navlog import NavLog
from .plotting import GRAVITY_MPS2

POS = [IDX_X, IDX_Y]


def frame_indices(
    time: np.ndarray, fps: float, speed: float = 1.0, start_time: float | None = None, end_time: float | None = None
) -> np.ndarray:
    """Map video frames to log step indices so the movie plays at `speed` x real time.

    Each frame shows the latest step at or before its playback time. When the log is
    sparser than the frame rate, consecutive frames repeat an index (keeping the pace).
    """
    if fps <= 0 or speed <= 0:
        raise ValueError("fps and speed must be positive")
    t0 = time[0] if start_time is None else max(start_time, time[0])
    t1 = time[-1] if end_time is None else min(end_time, time[-1])
    if t1 < t0:
        raise ValueError(f"empty time range [{t0}, {t1}]")
    targets = np.arange(t0, t1 + 1e-9, speed / fps)
    indices = np.clip(np.searchsorted(time, targets + 1e-9, side="right") - 1, 0, time.shape[0] - 1)
    last = int(np.searchsorted(time, t1 + 1e-9, side="right") - 1)
    if indices[-1] != last:
        indices = np.append(indices, last)
    return indices


class _Movie:
    """Shared frame timing and movie output; subclasses build `figure` and implement `update`."""

    figure: plt.Figure
    fps: float
    frame_indices: np.ndarray

    def update(self, frame: int) -> list:
        raise NotImplementedError

    def _freeze_layout(self) -> None:
        # Solve constrained layout once on the first frame, then freeze it: re-solving on every
        # frame dominates render time (and would make the axes jitter as tick labels change).
        self.update(0)
        self.figure.canvas.draw()
        self.figure.set_layout_engine("none")

    def make_animation(self) -> FuncAnimation:
        """Wrap `update` in a FuncAnimation (keep a reference to it until it is saved or shown)."""
        return FuncAnimation(self.figure, self.update, frames=len(self.frame_indices), interval=1000.0 / self.fps, blit=False, repeat=False)

    def save(self, path: str | Path, dpi: int = 100, progress_callback=None) -> Path:
        """Write the movie; `.gif` uses Pillow, anything else (e.g. `.mp4`) uses ffmpeg."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".gif":
            writer = PillowWriter(fps=self.fps)
        else:
            if not FFMpegWriter.isAvailable():
                raise RuntimeError("ffmpeg not found; install it or write a .gif instead")
            # H.264 + yuv420p for broad player support; that pixel format needs even frame dimensions.
            width, height = self.figure.get_size_inches() * dpi
            self.figure.set_size_inches(2 * round(width / 2) / dpi, 2 * round(height / 2) / dpi)
            writer = FFMpegWriter(fps=self.fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
        self.make_animation().save(path, writer=writer, dpi=dpi, progress_callback=progress_callback)
        return path


class NavAnimator(_Movie):
    def __init__(
        self,
        log: NavLog,
        fps: float = 30.0,
        speed: float = 1.0,
        confidence: float = 0.95,
        trail_seconds: float | None = None,
        follow_half_width: float | None = None,
        show_prior: bool = True,
        gnss_hold_seconds: float = 1.0,
        start_time: float | None = None,
        end_time: float | None = None,
        figsize: tuple[float, float] = (9.0, 9.0),
    ):
        self.log = log
        self.fps = fps
        self.confidence = confidence
        self.trail_seconds = trail_seconds
        self.follow_half_width = follow_half_width
        self.show_prior = show_prior
        self.gnss_hold_seconds = gnss_hold_seconds
        self.frame_indices = frame_indices(log.time, fps, speed, start_time, end_time)

        self._box_ratio = 1.0  # axes height / width in pixels, known once the layout is solved
        self._precompute()
        self.figure, (self.ax, self.panel_ax) = plt.subplots(
            2, 1, figsize=figsize, gridspec_kw={"height_ratios": [4, 1]}, constrained_layout=True
        )
        self._build_plane()
        self._build_panel()
        self._freeze_layout()
        extent = self.ax.get_window_extent()
        self._box_ratio = extent.height / extent.width
        if not self.follow_half_width:
            self._set_view(0.5 * (self.fixed_limits[0] + self.fixed_limits[1]), 0.5 * (self.fixed_limits[1] - self.fixed_limits[0]))
        self.update(0)

    # ------------------------------------------------------------------ setup

    def _precompute(self) -> None:
        log = self.log
        pos_cov = log.covariances[:, 0:2, 0:2]
        self.post_width, self.post_height, self.post_angle = covariance_ellipse(pos_cov, self.confidence)
        self.prior_width, self.prior_height, self.prior_angle = covariance_ellipse(log.prior_covariances[:, 0:2, 0:2], self.confidence)
        self.gnss_width, self.gnss_height, self.gnss_angle = covariance_ellipse(log.gnss_covariance[0:2, 0:2], self.confidence)
        self.yaw_half_width = normal_1d_scale(self.confidence) * np.sqrt(np.clip(log.covariances[:, IDX_YAW, IDX_YAW], 0.0, None))

        self.gnss_indices = np.flatnonzero(log.gnss_available)
        update_marks = np.where(log.gnss_available, np.arange(log.num_steps), -1)
        self.last_update_index = np.maximum.accumulate(update_marks)

        self.semi_major = 0.5 * self.post_width
        self.position_error = (
            np.linalg.norm(log.state_estimates[:, POS] - log.truth_states[:, POS], axis=1) if log.has_truth else None
        )

        i0, i1 = self.frame_indices[0], self.frame_indices[-1]
        points = [log.state_estimates[i0:i1 + 1, POS], log.gnss_measurements[i0:i1 + 1, 0:2]]
        if log.has_truth:
            points.append(log.truth_states[i0:i1 + 1, POS])
        points = np.concatenate(points)
        lower, upper = np.nanmin(points, axis=0), np.nanmax(points, axis=0)
        span = float(np.max(upper - lower)) or 1.0
        # Pad by a typical (not the initial, very large) ellipse size so most ellipses stay in view.
        pad = 0.05 * span + float(np.nanpercentile(self.semi_major[i0:i1 + 1], 90))
        self.fixed_limits = (lower - pad, upper + pad)
        view_size = 2.0 * self.follow_half_width if self.follow_half_width else span + 2.0 * pad
        self.heading_length = 0.06 * view_size

    def _build_plane(self) -> None:
        ax = self.ax
        percent = f"{100.0 * self.confidence:g}%"
        self.gnss_past = ax.scatter([], [], s=12, color="tab:red", alpha=0.3, label="GNSS fixes")
        if self.log.has_truth:
            (self.truth_line,) = ax.plot([], [], color="tab:orange", linewidth=1.5, label="truth")
            (self.truth_marker,) = ax.plot([], [], marker="x", markersize=8, color="tab:orange", linestyle="none")
        (self.est_line,) = ax.plot([], [], color="tab:blue", linewidth=1.5, label="EKF")
        self.post_ellipse = Ellipse(
            (0.0, 0.0), 0.0, 0.0, facecolor="tab:blue", edgecolor="tab:blue", alpha=0.25, label=f"EKF {percent} position"
        )
        self.prior_ellipse = Ellipse(
            (0.0, 0.0), 0.0, 0.0, fill=False, edgecolor="tab:blue", linestyle="--", linewidth=1.0, label=f"prior {percent} (last update)"
        )
        self.gnss_ellipse = Ellipse(
            (0.0, 0.0), self.gnss_width, self.gnss_height, angle=self.gnss_angle,
            fill=False, edgecolor="tab:red", linestyle=":", linewidth=1.2, label=f"GNSS {percent} (R)",
        )
        for patch in (self.post_ellipse, self.prior_ellipse, self.gnss_ellipse):
            ax.add_patch(patch)
        (self.gnss_current,) = ax.plot([], [], marker="o", markersize=7, color="tab:red", markeredgecolor="black", linestyle="none")
        (self.yaw_fan,) = ax.plot([], [], color="tab:blue", linewidth=0.8, alpha=0.6)
        (self.heading_line,) = ax.plot([], [], color="tab:blue", linewidth=2.0)
        (self.est_marker,) = ax.plot([], [], marker="o", markersize=5, color="tab:blue", linestyle="none")
        self.info_text = ax.text(
            0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left", family="monospace", fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True)
        ax.legend(loc="lower right", fontsize=8)

    def _build_panel(self) -> None:
        ax = self.panel_ax
        time = self.log.time
        i0, i1 = self.frame_indices[0], self.frame_indices[-1]
        (self.radius_line,) = ax.plot([], [], color="tab:blue", label=f"{100.0 * self.confidence:g}% ellipse semi-major axis")
        values = [self.semi_major[i0:i1 + 1]]
        if self.position_error is not None:
            (self.error_line,) = ax.plot([], [], color="tab:orange", label="|position error|")
            values.append(self.position_error[i0:i1 + 1])
        self.cursor = ax.axvline(time[i0], color="gray", linewidth=0.8)
        ymax = max(float(np.nanmax(v)) for v in values if np.isfinite(v).any())
        ax.set_xlim(time[i0], time[i1])
        ax.set_ylim(0.0, 1.1 * ymax if ymax > 0 else 1.0)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("[m]")
        ax.grid(True)
        ax.legend(loc="upper right", fontsize=8)

    # ----------------------------------------------------------------- update

    def update(self, frame: int) -> list:
        log = self.log
        i = int(self.frame_indices[frame])
        t = log.time[i]
        start = self.frame_indices[0]
        if self.trail_seconds is not None:
            start = max(start, int(np.searchsorted(log.time, t - self.trail_seconds)))

        x, y = log.state_estimates[i, POS]
        self.est_line.set_data(log.state_estimates[start:i + 1, IDX_X], log.state_estimates[start:i + 1, IDX_Y])
        self.est_marker.set_data([x], [y])
        if log.has_truth:
            self.truth_line.set_data(log.truth_states[start:i + 1, IDX_X], log.truth_states[start:i + 1, IDX_Y])
            self.truth_marker.set_data([log.truth_states[i, IDX_X]], [log.truth_states[i, IDX_Y]])

        past = self.gnss_indices[(self.gnss_indices >= start) & (self.gnss_indices <= i)]
        self.gnss_past.set_offsets(log.gnss_measurements[past, 0:2] if past.size else np.empty((0, 2)))

        self._set_ellipse(self.post_ellipse, (x, y), self.post_width[i], self.post_height[i], self.post_angle[i])

        last = int(self.last_update_index[i])
        age = t - log.time[last] if last >= 0 else np.inf
        recent_fix = age <= self.gnss_hold_seconds
        self.gnss_current.set_visible(recent_fix)
        self.gnss_ellipse.set_visible(recent_fix)
        self.prior_ellipse.set_visible(recent_fix and self.show_prior)
        if recent_fix:
            fix = log.gnss_measurements[last, 0:2]
            self.gnss_current.set_data([fix[0]], [fix[1]])
            self.gnss_ellipse.set_center(tuple(fix))
            prior_xy = tuple(log.prior_state_estimates[last, POS])
            self._set_ellipse(self.prior_ellipse, prior_xy, self.prior_width[last], self.prior_height[last], self.prior_angle[last])

        yaw = log.state_estimates[i, IDX_YAW]
        length = self.heading_length
        self.heading_line.set_data([x, x + length * np.cos(yaw)], [y, y + length * np.sin(yaw)])
        low, high = yaw - self.yaw_half_width[i], yaw + self.yaw_half_width[i]
        self.yaw_fan.set_data(
            [x, x + length * np.cos(low), np.nan, x, x + length * np.cos(high)],
            [y, y + length * np.sin(low), np.nan, y, y + length * np.sin(high)],
        )

        if self.follow_half_width:
            h = self.follow_half_width
            self._set_view((x, y), (h, h))

        self.info_text.set_text(self._info(i, last, age))

        self.radius_line.set_data(log.time[self.frame_indices[0]:i + 1], self.semi_major[self.frame_indices[0]:i + 1])
        if self.position_error is not None:
            self.error_line.set_data(log.time[self.frame_indices[0]:i + 1], self.position_error[self.frame_indices[0]:i + 1])
        self.cursor.set_xdata([t, t])
        return []

    def _set_view(self, center, half_extent) -> None:
        """Show at least `half_extent` around `center`, widening one axis to fill the axes box at 1:1 scale."""
        half_x, half_y = half_extent
        if half_y / half_x < self._box_ratio:
            half_y = half_x * self._box_ratio
        else:
            half_x = half_y / self._box_ratio
        self.ax.set_xlim(center[0] - half_x, center[0] + half_x)
        self.ax.set_ylim(center[1] - half_y, center[1] + half_y)

    @staticmethod
    def _set_ellipse(patch: Ellipse, center, width: float, height: float, angle: float) -> None:
        patch.set_center(center)
        patch.set_width(width)
        patch.set_height(height)
        patch.set_angle(angle)

    def _info(self, i: int, last: int, age: float) -> str:
        log = self.log
        if last < 0:
            gnss = "no fix yet"
        elif age == 0.0:
            gnss = "fix (update)"
        elif age <= self.gnss_hold_seconds:
            gnss = f"fix ({age:.1f} s ago)"
        else:
            gnss = f"outage {age:.1f} s"
        percent = f"{100.0 * self.confidence:g}%"
        lines = [
            f"t    = {log.time[i]:7.2f} s",
            f"GNSS : {gnss}",
            f"pos {percent}: {0.5 * self.post_width[i]:.2f} x {0.5 * self.post_height[i]:.2f} m (semi-axes)",
            f"yaw  = {np.rad2deg(log.state_estimates[i, IDX_YAW]):7.1f} ± {np.rad2deg(self.yaw_half_width[i]):.1f} deg",
        ]
        if self.position_error is not None and np.isfinite(self.position_error[i]):
            lines.append(f"|pos err| = {self.position_error[i]:.2f} m")
        return "\n".join(lines)


# (state index, panel title, unit, factor from SI to that unit)
BIAS_PANELS = (
    (IDX_BG, "gyro z bias", "deg/s", float(np.rad2deg(1.0))),
    (IDX_BAX, "accel x bias", "mG", 1000.0 / GRAVITY_MPS2),
    (IDX_BAY, "accel y bias", "mG", 1000.0 / GRAVITY_MPS2),
)


class BiasAnimator(_Movie):
    """Bias estimates vs. truth over a fixed time axis: one panel each for gyro z, accel x, accel y.

    Each panel reveals, up to the current time, the true bias, the estimate, and the
    estimate's `confidence` band (+/- k sigma from the covariance diagonal).
    """

    def __init__(
        self,
        log: NavLog,
        fps: float = 30.0,
        speed: float = 1.0,
        confidence: float = 0.95,
        start_time: float | None = None,
        end_time: float | None = None,
        figsize: tuple[float, float] = (9.0, 8.0),
    ):
        self.log = log
        self.fps = fps
        self.confidence = confidence
        self.frame_indices = frame_indices(log.time, fps, speed, start_time, end_time)
        k = normal_1d_scale(confidence)
        i0, i1 = self.frame_indices[0], self.frame_indices[-1]

        self.figure, axes = plt.subplots(len(BIAS_PANELS), 1, figsize=figsize, sharex=True, constrained_layout=True)
        self.title = self.figure.suptitle("")
        self.panels = []
        for ax, (index, title, unit, factor) in zip(axes, BIAS_PANELS):
            estimate = factor * log.state_estimates[:, index]
            half_band = k * factor * np.sqrt(np.clip(log.covariances[:, index, index], 0.0, None))
            truth = factor * log.truth_states[:, index] if log.has_truth else None

            band = Polygon(np.zeros((1, 2)), closed=True, facecolor="tab:blue", edgecolor="none", alpha=0.2,
                           label=f"{100.0 * confidence:g}% band")
            ax.add_patch(band)
            truth_line = ax.plot([], [], color="tab:orange", linewidth=2.0, label="true")[0] if truth is not None else None
            (estimate_line,) = ax.plot([], [], color="tab:blue", linewidth=1.5, label="estimate")
            cursor = ax.axvline(log.time[i0], color="gray", linewidth=0.8)
            text = ax.text(0.99, 0.95, "", transform=ax.transAxes, ha="right", va="top", family="monospace", fontsize=9,
                           bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})

            # Fix the y range to the values themselves (plus a typical band width), not the
            # very wide initial band, so convergence stays visible.
            shown = [estimate[i0:i1 + 1]] + ([truth[i0:i1 + 1]] if truth is not None else [])
            low = min(float(np.nanmin(v)) for v in shown)
            high = max(float(np.nanmax(v)) for v in shown)
            margin = max(0.1 * (high - low), float(np.nanmedian(half_band[i0:i1 + 1])), 1e-6)
            ax.set_ylim(low - margin, high + margin)
            ax.set_xlim(log.time[i0], log.time[i1])
            ax.set_title(title, loc="left", fontsize=10)
            ax.set_ylabel(f"[{unit}]")
            ax.grid(True)
            ax.legend(loc="upper left", fontsize=8, ncol=3)
            self.panels.append(
                {"estimate": estimate, "half_band": half_band, "truth": truth, "unit": unit,
                 "band": band, "truth_line": truth_line, "estimate_line": estimate_line, "cursor": cursor, "text": text}
            )
        axes[-1].set_xlabel("time [s]")
        self._freeze_layout()

    def update(self, frame: int) -> list:
        log = self.log
        i0 = int(self.frame_indices[0])
        i = int(self.frame_indices[frame])
        t = log.time[i]
        times = log.time[i0:i + 1]
        self.title.set_text(f"Bias estimates   t = {t:6.2f} s")
        for panel in self.panels:
            estimate = panel["estimate"][i0:i + 1]
            half_band = panel["half_band"][i0:i + 1]
            panel["estimate_line"].set_data(times, estimate)
            panel["band"].set_xy(np.concatenate([
                np.column_stack([times, estimate + half_band]),
                np.column_stack([times[::-1], (estimate - half_band)[::-1]]),
            ]))
            panel["cursor"].set_xdata([t, t])
            line = f"est {panel['estimate'][i]:+8.2f} ± {panel['half_band'][i]:.2f}"
            if panel["truth"] is not None:
                panel["truth_line"].set_data(times, panel["truth"][i0:i + 1])
                line += f"   true {panel['truth'][i]:+8.2f}"
            panel["text"].set_text(f"{line} {panel['unit']}")
        return []
