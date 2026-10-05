"""
Bayes estimation of a 1D Gaussian position: prior x ~ N(mu0, var0), observation z = x + w, w ~ N(0, r).

The posterior p(x | z) ∝ p(z | x) p(x) is again Gaussian (see doc/bayes_gaussian_1d.md):
    var1 = 1 / (1/var0 + 1/r)
    mu1  = var1 * (mu0/var0 + z/r)
which is the Kalman measurement update with H = 1:  K = var0 / (var0 + r),  mu1 = mu0 + K (z - mu0),  var1 = (1 - K) var0.

Run from the repo root (writes PNGs relative to the current directory):
    uv run python -m linear_kf.bayes_1d.bayes_gaussian_1d
    uv run python -m linear_kf.bayes_1d.bayes_gaussian_1d --prior-mean 2 --prior-std 1 --obs 3 --obs-std 0.5
"""

import argparse

import numpy as np


def gauss_pdf(x, mean, std):
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (np.sqrt(2.0 * np.pi) * std)


def gaussian_posterior(mu0, var0, z, r):
    """Posterior mean and variance of x given prior N(mu0, var0) and observation z = x + N(0, r)."""
    var1 = 1.0 / (1.0 / var0 + 1.0 / r)
    mu1 = var1 * (mu0 / var0 + z / r)
    return mu1, var1


def plot_bayes(mu0, std0, z, std_r, ofile):
    import matplotlib.pyplot as plt

    mu1, var1 = gaussian_posterior(mu0, std0 ** 2, z, std_r ** 2)
    std1 = np.sqrt(var1)
    K = std0 ** 2 / (std0 ** 2 + std_r ** 2)

    lo = min(mu0 - 4 * std0, z - 4 * std_r)
    hi = max(mu0 + 4 * std0, z + 4 * std_r)
    x = np.linspace(lo, hi, 1000)
    prior = gauss_pdf(x, mu0, std0)
    like = gauss_pdf(x, z, std_r)  # p(z | x) as a function of x (normalized over x, since it is symmetric in x and z)
    post = gauss_pdf(x, mu1, std1)
    # Numerical check: normalize prior x likelihood on the grid; it should coincide with the closed-form posterior.
    prod = prior * like
    prod_norm = prod / np.trapezoid(prod, x)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, prior, color='tab:blue', lw=2, label=rf'prior $p(x) = \mathcal{{N}}({mu0:g}, {std0:g}^2)$')
    ax.plot(x, like, color='tab:orange', lw=2, label=rf'likelihood $p(z{{=}}{z:g} \mid x) = \mathcal{{N}}(x; {z:g}, {std_r:g}^2)$')
    ax.plot(x, post, color='tab:red', lw=2.5, label=rf'posterior $p(x \mid z) = \mathcal{{N}}({mu1:.3g}, {std1:.3g}^2)$')
    ax.plot(x[::25], prod_norm[::25], 'o', color='k', ms=3, label=r'normalized $p(z \mid x)\,p(x)$ (numerical)')
    for m, c in [(mu0, 'tab:blue'), (z, 'tab:orange'), (mu1, 'tab:red')]:
        ax.axvline(m, color=c, ls='--', lw=1)
    ax.set_xlabel('position $x$ [m]')
    ax.set_ylabel('probability density')
    ax.set_xlim([lo, hi])
    ax.set_ylim(bottom=0)
    ax.grid(True)
    ax.legend(loc='upper left', fontsize='small')
    ax.text(0.99, 0.97,
            f'K = {std0:g}² / ({std0:g}² + {std_r:g}²) = {K:.3g}\n'
            f'mean: {mu0:g} + K ({z:g} - {mu0:g}) = {mu1:.3g} m\n'
            f'std:  √((1 - K) {std0:g}²) = {std1:.3g} m',
            transform=ax.transAxes, ha='right', va='top', family='monospace', fontsize='small',
            bbox={'facecolor': 'white', 'alpha': 0.85, 'lw': 0.5})
    fig.suptitle(f'Bayes estimation of 1D Gaussian position: prior {mu0:g} m, observation {z:g} m')
    fig.tight_layout()
    fig.savefig(ofile)
    print(ofile)


def plot_obs_noise_sweep(mu0, std0, z, std_rs, ofile):
    """Posterior for several observation noise levels: precise observations pull the estimate toward z."""
    import matplotlib.pyplot as plt

    lo = min(mu0, z) - 3.5 * std0
    hi = max(mu0, z) + 3.5 * std0
    x = np.linspace(lo, hi, 1000)
    fig, axes = plt.subplots(len(std_rs), 1, figsize=(9, 2.2 * len(std_rs)), sharex=True)
    for ax, std_r in zip(np.atleast_1d(axes), std_rs):
        mu1, var1 = gaussian_posterior(mu0, std0 ** 2, z, std_r ** 2)
        K = std0 ** 2 / (std0 ** 2 + std_r ** 2)
        ax.plot(x, gauss_pdf(x, mu0, std0), color='tab:blue', lw=1.5, label='prior')
        ax.plot(x, gauss_pdf(x, z, std_r), color='tab:orange', lw=1.5, label='likelihood')
        ax.plot(x, gauss_pdf(x, mu1, np.sqrt(var1)), color='tab:red', lw=2, label='posterior')
        ax.axvline(mu1, color='tab:red', ls='--', lw=1)
        ax.set_title(rf'$\sigma_z$ = {std_r:g} m: K = {K:.2f}, posterior $\mathcal{{N}}({mu1:.2f}, {np.sqrt(var1):.2f}^2)$', fontsize='medium')
        ax.set_ylabel('density')
        ax.set_ylim(bottom=0)
        ax.grid(True)
    np.atleast_1d(axes)[0].legend(loc='upper left', fontsize='small')
    np.atleast_1d(axes)[-1].set_xlabel('position $x$ [m]')
    np.atleast_1d(axes)[-1].set_xlim([lo, hi])
    fig.suptitle(f'Effect of observation noise: prior N({mu0:g}, {std0:g}²), observation {z:g} m')
    fig.tight_layout()
    fig.savefig(ofile)
    print(ofile)


def main():
    parser = argparse.ArgumentParser(description='Plot prior, likelihood and posterior of a 1D Gaussian position (Bayes update).')
    parser.add_argument('--prior-mean', type=float, default=2.0, help='prior mean [m] (default: %(default)s)')
    parser.add_argument('--prior-std', type=float, default=1.0, help='prior std [m] (default: %(default)s)')
    parser.add_argument('--obs', type=float, default=3.0, help='observation z [m] (default: %(default)s)')
    parser.add_argument('--obs-std', type=float, default=0.5, help='observation noise std [m] (default: %(default)s)')
    parser.add_argument('--sweep-obs-std', type=float, nargs='+', default=[2.0, 1.0, 0.5, 0.2],
                        help='observation noise stds for the sweep figure (default: %(default)s)')
    args = parser.parse_args()

    plot_bayes(args.prior_mean, args.prior_std, args.obs, args.obs_std, 'bayes_gaussian_1d.png')
    plot_obs_noise_sweep(args.prior_mean, args.prior_std, args.obs, args.sweep_obs_std, 'bayes_gaussian_1d_obs_noise.png')


if __name__ == '__main__':
    main()
