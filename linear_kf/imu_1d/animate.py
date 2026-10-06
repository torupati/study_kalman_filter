"""Movie of the 1D IMU + position KF: variance grows in predict, shrinks in update.

Between position fixes the accelerometer drives the prediction and the position
distribution spreads out (``P <- F P F^T + Q``, as in process_noise_sim). At each
fix the posterior replaces the prior, which is the 1D Bayes update of bayes_1d
(``K = P_pp / (P_pp + r)``) applied to the position marginal. By default the movie
runs on without stopping; with ``--update-pause SECONDS`` it stops at each fix while
the likelihood ``N(z, r)`` fades in on top of the prior and the prior morphs into
the posterior.

Movie panels:
  left   position density over absolute position x: prior (predicted), likelihood,
         posterior, and the true position drawn as a "car" box; the x window follows the car
  right  (position, velocity) error from the truth with the 2-sigma covariance ellipse;
         the update squeezes it along position, the prediction shears it
The position error and +-2 sigma over time (the sawtooth of sigma) is saved as a
separate image next to the movie (<output stem>_error.png).

    uv run python -m linear_kf.imu_1d.animate                     # ~20 s sim with a 9-15 s position outage
    uv run python -m linear_kf.imu_1d.animate --update-pause 1.5  # stop 1.5 s at each fix to show the Bayes update
    uv run python -m linear_kf.imu_1d.animate --output kf.gif --duration 8 --no-outage
"""

import argparse
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.artist import Artist
from matplotlib.axis import Axis
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PIL import Image

from .demo import P0, X0
from .kf import KalmanFilterImu1d, run_filter
from .plotting import EST, MEAS, TRUTH, _style
from .simulator import SimConfig, SimResult, simulate

PRIOR = EST
POST = "#d62728"


@dataclass
class Track:
    """Prior and posterior at every IMU sample (equal where there is no position update)."""

    x_prior: np.ndarray  # (N, 2)
    P_prior: np.ndarray  # (N, 2, 2)
    x_post: np.ndarray  # (N, 2)
    P_post: np.ndarray  # (N, 2, 2)


@dataclass
class Movie:
    """A figure, a per-frame update and the artists that update changes; rendered by blitting.

    Nearly all of a full redraw is text (titles, tick labels, legends, suptitle) that never changes, while the
    curves cost well under a millisecond. So the figure is drawn once without the changing artists and kept as
    the background; each frame restores it and redraws only the changing artists, plus the static artists that
    sit above them in the same axes (spines, legend), so the stacking order is that of a full redraw.
    """

    fig: Figure
    update: Callable[[int], None]
    n_frames: int
    dynamic: list[Artist]

    def _redraw_order(self) -> list[Artist]:
        dyn = set(self.dynamic)
        order = []
        for ax in self.fig.axes:
            children = ax.get_children()
            # lowest changing artist other than an axis (an axis' grid lies under everything anyway)
            floor = min((a.get_zorder() for a in children if a in dyn and not isinstance(a, Axis)), default=np.inf)
            order += sorted(  # stable, so equal zorders keep the order Axes.draw uses
                (a for a in children if a in dyn or (a.get_zorder() >= floor and a is not ax.patch and not isinstance(a, Axis))),
                key=lambda a: a.get_zorder(),
            )
        return order

    def frames(self, dpi: float) -> Iterator[memoryview]:
        """RGBA frame buffers; each is overwritten by the next frame, so copy it to keep it."""
        fig = self.fig
        fig.set_dpi(dpi)
        canvas = FigureCanvasAgg(fig)
        redraw = self._redraw_order()
        for a in redraw:
            a.set_animated(True)  # Axes.draw skips these
        try:
            self.update(0)
            canvas.draw()
            background = canvas.copy_from_bbox(fig.bbox)
            for i in range(self.n_frames):
                self.update(i)
                canvas.restore_region(background)
                for a in redraw:
                    fig.draw_artist(a)
                yield canvas.buffer_rgba()
        finally:
            for a in redraw:
                a.set_animated(False)

    def save(self, out: Path, fps: float, dpi: float, progress_callback: Callable[[int, int], None] | None = None):
        """Write a .gif (Pillow) or, for any other suffix, an H.264 movie (ffmpeg)."""
        frames = self.frames(dpi)
        n = self.n_frames
        if Path(out).suffix.lower() == ".gif":
            images = []
            for i, buf in enumerate(frames):
                h, w = buf.shape[:2]
                images.append(Image.frombuffer("RGBA", (w, h), bytes(buf), "raw", "RGBA", 0, 1))
                if progress_callback:
                    progress_callback(i, n)
            images[0].save(out, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0)
            return
        if not FFMpegWriter.isAvailable():
            raise RuntimeError("ffmpeg not found; install it or write a .gif instead")
        proc = None
        try:
            for i, buf in enumerate(frames):
                if proc is None:  # the frame size is known once the first frame is drawn
                    proc = subprocess.Popen(_ffmpeg_cmd(buf.shape[1], buf.shape[0], fps, out), stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                proc.stdin.write(buf)
                if progress_callback:
                    progress_callback(i, n)
        finally:
            if proc is not None:
                _, err = proc.communicate()
        if proc is not None and proc.returncode:
            raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {err.decode(errors='replace').strip()}")


def _ffmpeg_cmd(w: int, h: int, fps: float, out: Path) -> list[str]:
    return [FFMpegWriter.bin_path(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}",
            "-framerate", f"{fps:g}", "-i", "pipe:",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # libx264 + yuv420p needs even width/height
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)]


def run_track(sim: SimResult) -> Track:
    cfg = sim.config
    kf = KalmanFilterImu1d(cfg.accel_noise_std, cfg.pos_noise_std)
    res = run_filter(kf, sim.t, sim.acc_meas, sim.pos_meas, X0, P0)
    x_prior, P_prior = res.x.copy(), res.P.copy()
    x_prior[0], P_prior[0] = X0, P0
    for k in range(1, len(sim.t)):
        x_prior[k], P_prior[k] = kf.predict(res.x[k - 1], res.P[k - 1], sim.acc_meas[k - 1], sim.t[k] - sim.t[k - 1])
    return Track(x_prior, P_prior, res.x, res.P)


def frame_plan(t: np.ndarray, pos_available: np.ndarray, fps: float, speed: float, update_pause: float) -> list[tuple[str, int, float]]:
    """Frames as (stage, IMU index k, progress s in [0, 1]).

    "predict" frames step through the IMU samples at `speed` x real time. With
    `update_pause` > 0, every position update k is expanded into "likelihood"
    (fade in), "blend" (prior -> posterior) and "hold" frames, together lasting
    `update_pause` seconds; with 0 the movie does not stop, and the update is a
    single "hold" frame (prior, likelihood and posterior) on the time line.
    """
    dt = t[1] - t[0]
    upd = set(np.flatnonzero(pos_available).tolist())
    ks = np.unique(np.round(np.arange(0.0, t[-1] - t[0] + 1e-9, speed / fps) / dt).astype(int))
    ks = np.union1d(ks, list(upd))  # never skip an update sample
    n_pause = max(3, int(round(update_pause * fps)))
    n_like, n_blend = int(round(0.3 * n_pause)), int(round(0.45 * n_pause))
    n_hold = n_pause - n_like - n_blend
    plan = []
    for k in ks:
        k = int(k)
        if k in upd and update_pause <= 0:
            plan.append(("hold", k, 1.0))
        elif k in upd:
            plan += [("likelihood", k, (i + 1) / n_like) for i in range(n_like)]
            plan += [("blend", k, (i + 1) / n_blend) for i in range(n_blend)]
            plan += [("hold", k, 1.0)] * n_hold
        else:
            plan.append(("predict", k, 1.0))
    return plan


def gauss_pdf(x, mean, std):
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (np.sqrt(2.0 * np.pi) * std)


def ellipse_xy(mean: np.ndarray, P: np.ndarray, n_sigma: float = 2.0, n: int = 100) -> np.ndarray:
    """Points of the n_sigma contour of N(mean, P), shape (n, 2)."""
    w, V = np.linalg.eigh(P)
    a = np.linspace(0.0, 2.0 * np.pi, n)
    return mean + n_sigma * (np.c_[np.cos(a), np.sin(a)] * np.sqrt(np.maximum(w, 0.0))) @ V.T


def error_history(trk: Track, sim: SimResult) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(t, position error, sigma_p) with both the prior and the posterior point at each update time (the sawtooth)."""
    upd = sim.pos_available
    e_prior, e_post = trk.x_prior[:, 0] - sim.pos_true, trk.x_post[:, 0] - sim.pos_true
    s_prior, s_post = np.sqrt(trk.P_prior[:, 0, 0]), np.sqrt(trk.P_post[:, 0, 0])
    idx = np.flatnonzero(upd)
    order = np.argsort(np.r_[np.arange(len(sim.t)), idx - 0.5], kind="stable")  # each prior just before its posterior
    return (np.r_[sim.t, sim.t[idx]][order], np.r_[e_post, e_prior[idx]][order], np.r_[s_post, s_prior[idx]][order])


def plot_error_history(sim: SimResult, trk: Track):
    """Position error and +-2 sigma over time: sigma grows in predict and drops at every position update."""
    t, upd = sim.t, sim.pos_available
    ht, he, hs = error_history(trk, sim)
    fig, ax = plt.subplots(figsize=(11, 3.8), layout="constrained")
    if sim.config.pos_outage is not None:
        ax.axvspan(*sim.config.pos_outage, color="#999999", alpha=0.15, lw=0, label="no position")
    ax.fill_between(ht, -2 * hs, 2 * hs, color=PRIOR, alpha=0.15, lw=0)
    ax.plot(ht, 2 * hs, color=PRIOR, lw=1.4, label=r"$\pm 2\sigma_p$")
    ax.plot(ht, -2 * hs, color=PRIOR, lw=1.4)
    ax.plot(ht, he, color=TRUTH, lw=1.2, label="KF position error")
    ax.plot(t[upd], sim.pos_meas[upd] - sim.pos_true[upd], ".", color=MEAS, ms=5, label="measurement error")
    ax.axhline(0.0, color=TRUTH, lw=0.6)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("time [s]")
    ax.set_ylabel("position error [m]")
    _style(ax)
    ax.legend(loc="upper right", fontsize=8, frameon=False, ncols=4)
    ax.set_title("1D IMU + position KF: position error and its $\\pm 2\\sigma$ (prior and posterior at each update)")
    return fig


def make_animation(sim: SimResult, trk: Track, fps: float, speed: float, update_pause: float) -> Movie:
    cfg = sim.config
    t, upd = sim.t, sim.pos_available
    r_std = cfg.pos_noise_std
    p_true = sim.pos_true
    truth = np.c_[p_true, sim.vel_true]
    e_prior, e_post = trk.x_prior - truth, trk.x_post - truth
    sp_prior, sp_post = np.sqrt(trk.P_prior[:, 0, 0]), np.sqrt(trk.P_post[:, 0, 0])
    z, z_err = sim.pos_meas, sim.pos_meas - p_true
    plan = frame_plan(t, upd, fps, speed, update_pause)

    # Fixed axis ranges, from the widest prior after the first two fixes; the large initial P0 (and its first
    # prediction) are clipped, otherwise the later, much smaller distributions would be hard to see.
    after0 = t >= t[0] + 1.5 / cfg.pos_rate
    if not after0.any():  # run shorter than that: use everything
        after0 = np.ones_like(t, dtype=bool)
    p_lim = 1.15 * max(np.max(2.0 * sp_prior[after0] + np.abs(e_prior[after0, 0])), 4.0 * r_std)
    v_lim = 1.3 * np.max(2.0 * np.sqrt(trk.P_prior[after0, 1, 1]) + np.abs(e_prior[after0, 1]))
    pdf_top = 1.15 * max(gauss_pdf(0.0, 0.0, sp_post[after0].min()), gauss_pdf(0.0, 0.0, r_std))
    dx = np.linspace(-p_lim, p_lim, 600)  # density grid relative to the car; the window follows the car

    fig = plt.figure(figsize=(12, 4.8), layout="constrained")
    gs = fig.add_gridspec(1, 3)
    ax_pdf = fig.add_subplot(gs[0, :2])
    ax_ell = fig.add_subplot(gs[0, 2])
    for ax in (ax_pdf, ax_ell):
        _style(ax)

    # Position density in absolute position, with the true position drawn as a car.
    (l_prior,) = ax_pdf.plot(dx, 0 * dx, color=PRIOR, lw=2.2, label="prior (predicted) $p(x)$")
    (l_like,) = ax_pdf.plot(dx, 0 * dx, color=MEAS, lw=2.0, label=r"likelihood $p(z \mid x)$")
    (l_post,) = ax_pdf.plot(dx, 0 * dx, color=POST, lw=2.5, label=r"posterior $p(x \mid z)$")
    m_z = ax_pdf.axvline(0.0, color=MEAS, ls="--", lw=1.0)
    m_est = ax_pdf.axvline(0.0, color=PRIOR, ls="--", lw=1.0)
    m_true = ax_pdf.axvline(0.0, color=TRUTH, lw=1.0, label="true position")
    car_w = 0.12 * p_lim  # [m] drawn width; x in data, y in axes coordinates
    car = Rectangle((0.0, 0.02), car_w, 0.09, transform=ax_pdf.get_xaxis_transform(), facecolor="#f2c14e",
                    edgecolor=TRUTH, lw=1.2, zorder=5, label="car (truth)")
    ax_pdf.add_patch(car)
    car_text = ax_pdf.text(0.0, 0.065, "car", transform=ax_pdf.get_xaxis_transform(), ha="center", va="center",
                           fontsize=9, fontweight="bold", zorder=6)
    ax_pdf.set_ylim(0.0, pdf_top)
    ax_pdf.set_xlabel("position $x$ [m]")
    ax_pdf.set_ylabel("density")
    ax_pdf.legend(loc="upper right", fontsize=8, frameon=False)
    phase = ax_pdf.set_title(" ", loc="left", fontsize=11, fontweight="bold")
    info = ax_pdf.text(0.01, 0.97, "", transform=ax_pdf.transAxes, va="top", family="monospace", fontsize=8.5)

    # (p, v) covariance ellipse, as errors from the truth.
    z_band = ax_ell.axvspan(0.0, 0.0, color=MEAS, alpha=0.15, lw=0, label=r"$z \pm 2\sigma_z$")
    (e_pr,) = ax_ell.plot([], [], color=PRIOR, lw=1.8, label=r"prior $2\sigma$")
    (e_po,) = ax_ell.plot([], [], color=POST, lw=2.2, label=r"posterior $2\sigma$")
    (c_pr,) = ax_ell.plot([], [], "o", color=PRIOR, ms=4)
    (c_po,) = ax_ell.plot([], [], "o", color=POST, ms=4)
    ax_ell.plot([0.0], [0.0], "+", color=TRUTH, ms=12, mew=1.5)
    ax_ell.set_xlim(-p_lim, p_lim)
    ax_ell.set_ylim(-v_lim, v_lim)
    ax_ell.set_xlabel("position error [m]")
    ax_ell.set_ylabel("velocity error [m/s]")
    ax_ell.legend(loc="upper right", fontsize=7, frameon=False)
    fig.suptitle(
        f"1D Kalman filter: accelerometer prediction ({cfg.imu_rate:g} Hz) + position update ({cfg.pos_rate:g} Hz, "
        rf"$\sigma_z$ = {r_std:g} m)"
    )

    def update(i):
        stage, k, s = plan[i]
        ep, Pp = e_prior[k], trk.P_prior[k]
        if stage == "predict":
            # Prior == posterior here; draw it as the (growing) prior.
            ep, Pp = e_post[k], trk.P_post[k]
            mean, P, show_post = ep, Pp, False
        elif stage == "likelihood":
            mean, P, show_post = ep, Pp, False
        elif stage == "blend":
            a = 0.5 - 0.5 * np.cos(np.pi * s)  # ease in/out
            mean, P, show_post = (1 - a) * ep + a * e_post[k], (1 - a) * Pp + a * trk.P_post[k], True
        else:
            mean, P, show_post = e_post[k], trk.P_post[k], True
        is_upd = stage != "predict"
        pk = p_true[k]
        xs = pk + dx

        # Density panel. The prior is solid while predicting and fades once the posterior takes over.
        ax_pdf.set_xlim(pk - p_lim, pk + p_lim)
        l_prior.set_data(xs, gauss_pdf(dx, ep[0], np.sqrt(Pp[0, 0])))
        l_prior.set_alpha(0.35 if show_post else 1.0)
        l_prior.set_linestyle("--" if show_post else "-")
        m_est.set_xdata([pk + ep[0]] * 2)
        m_true.set_xdata([pk] * 2)
        car.set_x(pk - 0.5 * car_w)
        car_text.set_x(pk)
        like_a = s if stage == "likelihood" else 1.0
        l_like.set_visible(is_upd)
        m_z.set_visible(is_upd)
        if is_upd:
            l_like.set_data(xs, gauss_pdf(dx, z_err[k], r_std))
            l_like.set_alpha(like_a)
            m_z.set_xdata([z[k]] * 2)
        l_post.set_visible(show_post)
        if show_post:
            l_post.set_data(xs, gauss_pdf(dx, mean[0], np.sqrt(P[0, 0])))

        if is_upd:
            K = Pp[0, 0] / (Pp[0, 0] + r_std**2)
            phase.set_text(r"UPDATE (position fix $z$): prior $\times$ likelihood $\to$ posterior, variance shrinks")
            phase.set_color(POST)
            info.set_text(
                f"t = {t[k]:6.2f} s   x_true = {pk:6.2f} m\n"
                f"prior  {pk + ep[0]:6.2f} m  sigma {np.sqrt(Pp[0, 0]):5.3f} m\n"
                f"z      {z[k]:6.2f} m  sigma {r_std:5.3f} m\n"
                f"K_p = {K:5.3f}\n"
                f"post   {trk.x_post[k, 0]:6.2f} m  sigma {sp_post[k]:5.3f} m" + ("" if stage == "likelihood" else "  <-")
            )
        else:
            k_last = np.flatnonzero(upd[: k + 1])
            since = t[k] - t[k_last[-1]] if k_last.size else t[k]
            phase.set_text(r"PREDICT (accelerometer): $P \leftarrow F P F^\top + Q$, variance grows")
            phase.set_color(PRIOR)
            info.set_text(
                f"t = {t[k]:6.2f} s   x_true = {pk:6.2f} m\n"
                f"since last fix {since:5.2f} s\n"
                f"estimate {pk + ep[0]:6.2f} m\n"
                f"sigma_p  {np.sqrt(Pp[0, 0]):6.3f} m\n"
                f"sigma_v  {np.sqrt(Pp[1, 1]):6.3f} m/s"
            )

        # Ellipse panel.
        e_pr.set_data(*ellipse_xy(ep, Pp).T)
        c_pr.set_data([ep[0]], [ep[1]])
        e_pr.set_alpha(0.4 if show_post else 1.0)
        e_pr.set_linestyle("--" if show_post else "-")
        e_po.set_visible(show_post)
        c_po.set_visible(show_post)
        if show_post:
            e_po.set_data(*ellipse_xy(mean, P).T)
            c_po.set_data([mean[0]], [mean[1]])
        z_band.set_visible(is_upd)
        if is_upd:
            z_band.set_x(z_err[k] - 2 * r_std)
            z_band.set_width(4 * r_std)
            z_band.set_alpha(0.15 * like_a)

    update(0)
    fig.get_layout_engine().execute(fig)
    fig.set_layout_engine("none")  # freeze the layout: re-solving it every frame is slow and makes the axes jitter
    dynamic = [ax_pdf.xaxis, l_prior, l_like, l_post, m_z, m_est, m_true, car, car_text, phase, info,  # xlim follows the car
               z_band, e_pr, e_po, c_pr, c_po]
    return Movie(fig, update, len(plan), dynamic)


def progress_printer(every: int = 10):
    """Return a ``Movie.save`` progress_callback printing frame count, elapsed time and ETA on one line (stderr)."""
    t0 = time.monotonic()

    def callback(i: int, n: int):
        done = i + 1
        if done % every and done != n:
            return
        elapsed = time.monotonic() - t0
        eta = elapsed / done * (n - done)
        sys.stderr.write(f"\rrendering frame {done}/{n} ({100 * done / n:5.1f}%)  elapsed {elapsed:6.0f} s  ETA {eta:6.0f} s ")
        if done == n:
            sys.stderr.write("\n")
        sys.stderr.flush()

    return callback


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = SimConfig()
    p.add_argument("--output", type=Path, default=Path("outputs/linear_kf/imu_1d/kf_imu1d_anim.mp4"),
                   help="output file; .gif uses Pillow, otherwise ffmpeg (default: %(default)s)")
    p.add_argument("--duration", type=float, default=20.0, help="simulated time [s] (default: %(default)s)")
    p.add_argument("--imu-rate", type=float, default=d.imu_rate)
    p.add_argument("--pos-rate", type=float, default=d.pos_rate)
    p.add_argument("--accel-noise-density", type=float, default=d.accel_noise_density, help="m/s^2/sqrt(Hz)")
    p.add_argument("--pos-noise-std", type=float, default=d.pos_noise_std, help="m")
    p.add_argument("--outage", type=float, nargs=2, metavar=("START", "END"), default=(9.0, 15.0),
                   help="no position measurements in [START, END) s (default: %(default)s)")
    p.add_argument("--no-outage", action="store_true", help="position measurements throughout")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=float, default=30.0, help="movie frame rate (default: %(default)s)")
    p.add_argument("--speed", type=float, default=1.0, help="playback speed of the prediction vs. real time (default: %(default)s)")
    p.add_argument("--update-pause", type=float, default=0.0,
                   help="movie seconds to stop at each position update and animate the Bayes update; 0: no stop (default: %(default)s)")
    p.add_argument("--dpi", type=int, default=100)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = SimConfig(
        duration=args.duration,
        imu_rate=args.imu_rate,
        pos_rate=args.pos_rate,
        accel_noise_density=args.accel_noise_density,
        pos_noise_std=args.pos_noise_std,
        pos_outage=None if args.no_outage else tuple(args.outage),
    )
    sim = simulate(cfg, np.random.default_rng(args.seed))
    trk = run_track(sim)

    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    png = out.with_name(out.stem + "_error.png")
    err_fig = plot_error_history(sim, trk)
    err_fig.savefig(png, dpi=120)
    plt.close(err_fig)
    print(png)

    if out.suffix.lower() != ".gif" and not FFMpegWriter.isAvailable():
        raise SystemExit("ffmpeg not found; install it or write a .gif instead (--output xxx.gif)")
    movie = make_animation(sim, trk, args.fps, args.speed, args.update_pause)
    movie.save(out, args.fps, args.dpi, progress_callback=progress_printer())
    plt.close(movie.fig)
    print(out)


if __name__ == "__main__":
    main()
