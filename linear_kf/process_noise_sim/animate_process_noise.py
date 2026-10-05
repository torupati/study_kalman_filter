"""
Animate how the distributions of the random acceleration model (process_noise.py) evolve over time.

Three panels whose x axes are position, velocity and acceleration. Each frame shows, at time t,
a histogram of the N simulated samples and the analytic Gaussian density
    a ~ N(0, sigma^2 / dt)   (white, does not change with t)
    v ~ N(0, sigma^2 t)
    p ~ N(0, sigma^2 t^3 / 3)
so the velocity and position distributions visibly spread out as t grows.
The discretization is the same as process_noise.py: acc_k = sig0 * w_k / sqrt(dt), v += acc * dt, p += v * dt.

Run from the repo root (writes the movie relative to the current directory):
    uv run python -m linear_kf.process_noise_sim.animate_process_noise
    uv run python -m linear_kf.process_noise_sim.animate_process_noise --output pn.gif --speed 0.5
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

KEYS = ['pos', 'vel', 'acc']
XLABELS = ['position $p$ [m]', 'velocity $v$ [m/s]', 'acceleration $a$ [m/s$^2$]']
SYMBOLS = ['p', 'v', 'a']


def simulate(sig0, Fs, t_end, N, rng):
    """Return time (T,) and {'acc', 'vel', 'pos'} arrays of shape (N, T), starting from rest at the origin."""
    dt = 1.0 / Fs
    t = np.arange(int(round(t_end * Fs)) + 1) * dt
    acc = sig0 * rng.standard_normal((N, t.size)) / np.sqrt(dt)
    vel = np.cumsum(acc, axis=1) * dt
    pos = np.cumsum(vel, axis=1) * dt
    return t, {'acc': acc, 'vel': vel, 'pos': pos}


def analytic_std(t, sig0, dt):
    """Std of the random acceleration model: acc = sigma/sqrt(dt) (white), Var[v] = sigma^2 t, Var[p] = sigma^2 t^3 / 3."""
    return {
        'acc': np.full_like(t, sig0 / np.sqrt(dt)),
        'vel': sig0 * np.sqrt(t),
        'pos': sig0 * np.sqrt(np.power(t, 3) / 3.0),
    }


def gauss_pdf(x, std):
    return np.exp(-0.5 * (x / std) ** 2) / (np.sqrt(2.0 * np.pi) * std)


def make_animation(t, data, sig0, Fs, highlight, fps, speed, n_bins, ylim_scale):
    dt = 1.0 / Fs
    N = data['acc'].shape[0]
    stds = analytic_std(t, sig0, dt)
    # One frame per 1/fps of playback (the movie runs at `speed` x real time); skip t = 0, where p and v are exactly 0.
    frames = np.unique(np.clip(np.round(np.arange(0.0, t[-1] + 1e-9, speed / fps) / dt).astype(int), 1, t.size - 1))
    if frames[-1] != t.size - 1:
        frames = np.append(frames, t.size - 1)

    fig, axes = plt.subplots(3, 1, figsize=(9, 8))
    title = (f'Random acceleration model: distribution over N={N} sample paths vs. Gaussian\n'
             rf'noise density $\sigma$ = {sig0} m/s$^{{1.5}}$ (m/s$^2/\sqrt{{\mathrm{{Hz}}}}$), $F_s$ = {Fs} Hz')
    suptitle = fig.suptitle(title + f'\nt = {t[frames[0]]:.2f} s')
    artists = []
    for a, key, xlabel, sym in zip(axes, KEYS, XLABELS, SYMBOLS):
        # Fixed symmetric x range (4 sigma at the final time) and fixed bins, so the spreading is visible.
        xmax = 4.0 * stds[key][-1]
        x = np.linspace(-xmax, xmax, 400)
        edges = np.linspace(-xmax, xmax, n_bins + 1)
        hist = a.stairs(np.zeros(n_bins), edges, fill=True, color='0.6', alpha=0.6, label=f'samples (N={N})')
        pdf = a.plot(x, np.zeros_like(x), color='tab:blue', lw=2.0, label='Gaussian $\\mathcal{N}(0, \\sigma_' + sym + '^2(t))$')[0]
        band = a.axvspan(0.0, 0.0, color='tab:blue', alpha=0.12, lw=0, label=r'$\pm 2\sigma$')
        red = a.plot([], [], 'v', color='r', ms=9, label=f'sample #{highlight}')[0]
        text = a.text(0.01, 0.95, '', transform=a.transAxes, va='top', family='monospace')
        artists.append((x, edges, hist, pdf, band, red, text))
        # Fixed y range: `ylim_scale` x the density peak at the final time. Earlier, narrower distributions are
        # taller and get clipped; they spread into the window as t grows.
        a.set_ylim([0.0, ylim_scale * gauss_pdf(0.0, stds[key][-1])])
        a.set_xlim([-xmax, xmax])
        a.set_xlabel(xlabel)
        a.set_ylabel('density')
        a.grid(True)
    axes[0].legend(loc='upper right', fontsize='small')
    fig.tight_layout()

    def update(frame):
        k = frames[frame]
        for key, (x, edges, hist, pdf, band, red, text) in zip(KEYS, artists):
            s = stds[key][k]
            samples = data[key][:, k]
            counts, _ = np.histogram(samples, bins=edges)
            hist.set_data(counts / (N * np.diff(edges)))
            pdf.set_ydata(gauss_pdf(x, s))
            band.set_x(-2.0 * s)
            band.set_width(4.0 * s)
            ytop = band.axes.get_ylim()[1]
            red.set_data([samples[highlight]], [0.03 * ytop])
            text.set_text(f'analytic std {s:8.4f}\nsample   std {np.std(samples, ddof=1):8.4f}')
        suptitle.set_text(title + f'\nt = {t[k]:.2f} s')
        return []

    return fig, FuncAnimation(fig, update, frames=len(frames), interval=1000.0 / fps, blit=False, repeat=False)


def main():
    parser = argparse.ArgumentParser(description='Animate the distributions of the random acceleration model (process_noise.py) over time.')
    parser.add_argument('--output', default='process_noise_anim.mp4',
                        help='output file; .gif uses Pillow, otherwise ffmpeg (default: %(default)s)')
    parser.add_argument('--sig0', type=float, default=0.15, help='noise density sigma [m/s^1.5] (default: %(default)s)')
    parser.add_argument('--fs', type=float, default=200, help='sampling rate [Hz] (default: %(default)s)')
    parser.add_argument('--t-end', type=float, default=3.0, help='simulated duration [s] (default: %(default)s)')
    parser.add_argument('--num-samples', type=int, default=5000, help='number of simulated paths N (default: %(default)s)')
    parser.add_argument('--highlight', type=int, default=0, help='index of the path marked in red (default: %(default)s)')
    parser.add_argument('--bins', type=int, default=80, help='histogram bins per panel (default: %(default)s)')
    parser.add_argument('--ylim-scale', type=float, default=3.0,
                        help='y range = this x the final density peak; earlier, taller densities are clipped (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: %(default)s)')
    parser.add_argument('--fps', type=float, default=30, help='movie frame rate (default: %(default)s)')
    parser.add_argument('--speed', type=float, default=0.25, help='playback speed vs. real time (default: %(default)s)')
    parser.add_argument('--dpi', type=int, default=100, help='output resolution (default: %(default)s)')
    args = parser.parse_args()
    if not 0 <= args.highlight < args.num_samples:
        parser.error('need 0 <= --highlight < --num-samples')

    rng = np.random.default_rng(args.seed)
    t, data = simulate(args.sig0, args.fs, args.t_end, args.num_samples, rng)
    fig, anim = make_animation(t, data, args.sig0, args.fs, args.highlight, args.fps, args.speed, args.bins, args.ylim_scale)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == '.gif':
        writer = PillowWriter(fps=args.fps)
    else:
        if not FFMpegWriter.isAvailable():
            raise SystemExit('ffmpeg not found; install it or write a .gif instead (--output xxx.gif)')
        writer = FFMpegWriter(fps=args.fps, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
    anim.save(out, writer=writer, dpi=args.dpi)
    print(out)


if __name__ == '__main__':
    main()
