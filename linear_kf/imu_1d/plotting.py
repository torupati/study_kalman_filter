"""Matplotlib figure builders for the 1D IMU + position Kalman filter."""

import matplotlib.pyplot as plt
import numpy as np

from .kf import FilterResult
from .simulator import SimResult

TRUTH = "#444444"
EST = "#2a78d6"
MEAS = "#eb6834"
OTHER = "#1baf7a"
BAND = "#2a78d6"


def _style(ax):
    ax.grid(True, color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _shade_outage(ax, sim: SimResult):
    if sim.config.pos_outage is not None:
        ax.axvspan(*sim.config.pos_outage, color="#999999", alpha=0.15, linewidth=0, label="no position")


def plot_overview(sim: SimResult, title: str = ""):
    """Truth and measurements: acceleration, velocity, position."""
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True, layout="constrained")
    m = sim.pos_available
    axes[0].plot(sim.t, sim.acc_meas, color=MEAS, linewidth=0.5, alpha=0.6, label="accelerometer")
    axes[0].plot(sim.t, sim.acc_true, color=TRUTH, linewidth=1.5, label="true")
    axes[0].set_ylabel("acceleration [m/s$^2$]")
    axes[1].plot(sim.t, sim.vel_true, color=TRUTH, linewidth=1.5, label="true")
    axes[1].set_ylabel("velocity [m/s]")
    axes[2].plot(sim.t, sim.pos_true, color=TRUTH, linewidth=1.5, label="true")
    axes[2].plot(sim.t[m], sim.pos_meas[m], "o", color=MEAS, markersize=3, label="position measurement")
    axes[2].set_ylabel("position [m]")
    axes[2].set_xlabel("time [s]")
    for ax in axes:
        _style(ax)
        _shade_outage(ax, sim)
        ax.legend(loc="upper left", fontsize=8, frameon=False)
    if title:
        fig.suptitle(title)
    return fig


def plot_errors(sim: SimResult, res: FilterResult, title: str = ""):
    """Estimation error with the filter's own +-2 sigma band."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True, layout="constrained")
    m = sim.pos_available
    errs = (res.x[:, 0] - sim.pos_true, res.x[:, 1] - sim.vel_true)
    sigmas = (np.sqrt(res.P[:, 0, 0]), np.sqrt(res.P[:, 1, 1]))
    labels = ("position error [m]", "velocity error [m/s]")
    for i, ax in enumerate(axes):
        ax.fill_between(sim.t, -2 * sigmas[i], 2 * sigmas[i], color=BAND, alpha=0.15, linewidth=0, label=r"$\pm 2\sigma$ (filter)")
        if i == 0:
            ax.plot(sim.t[m], sim.pos_meas[m] - sim.pos_true[m], ".", color=MEAS, markersize=4, label="measurement error")
        ax.plot(sim.t, errs[i], color=EST, linewidth=1.2, label="KF error")
        ax.axhline(0.0, color=TRUTH, linewidth=0.8)
        ax.set_ylabel(labels[i])
        _style(ax)
        _shade_outage(ax, sim)
        ax.legend(loc="upper right", fontsize=8, frameon=False, ncols=2)
    axes[-1].set_xlabel("time [s]")
    if title:
        fig.suptitle(title)
    return fig


def plot_sigma_zoom(sim: SimResult, res: FilterResult, t_range: tuple[float, float], title: str = ""):
    """Position and velocity std over a short window: growth by prediction, drop at each update."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True, layout="constrained")
    w = (sim.t >= t_range[0]) & (sim.t <= t_range[1])
    upd = w & sim.pos_available
    for i, (ax, label) in enumerate(zip(axes, ("position std [m]", "velocity std [m/s]"))):
        sig = np.sqrt(res.P[:, i, i])
        ax.plot(sim.t[w], sig[w], color=EST, linewidth=1.5, label=r"$\sigma$ after each step")
        ax.plot(sim.t[upd], sig[upd], "o", color=MEAS, markersize=5, label="right after a position update")
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0.0)
        _style(ax)
        ax.legend(loc="lower right", fontsize=8, frameon=False)
    axes[-1].set_xlabel("time [s]")
    if title:
        fig.suptitle(title)
    return fig


def plot_compare_velocity(sim: SimResult, results: dict[str, FilterResult], title: str = ""):
    """Velocity error of several filters on the same data (one panel each, shared y)."""
    fig, axes = plt.subplots(len(results), 1, figsize=(8, 2.6 * len(results)), sharex=True, sharey=True, layout="constrained")
    for ax, (name, res), color in zip(np.atleast_1d(axes), results.items(), (EST, OTHER, MEAS)):
        sig = np.sqrt(res.P[:, 1, 1])
        ax.fill_between(sim.t, -2 * sig, 2 * sig, color=color, alpha=0.15, linewidth=0, label=r"$\pm 2\sigma$ (filter)")
        ax.plot(sim.t, res.x[:, 1] - sim.vel_true, color=color, linewidth=1.2, label="velocity error")
        ax.axhline(0.0, color=TRUTH, linewidth=0.8)
        ax.set_title(name, fontsize=10, loc="left")
        ax.set_ylabel("[m/s]")
        _style(ax)
        _shade_outage(ax, sim)
        ax.legend(loc="upper right", fontsize=8, frameon=False, ncols=2)
    np.atleast_1d(axes)[-1].set_xlabel("time [s]")
    if title:
        fig.suptitle(title)
    return fig


def plot_innovation(sim: SimResult, results: dict[str, FilterResult], title: str = ""):
    """Normalized innovation e / sqrt(S) at each position update; +-2 should hold ~95% of points."""
    fig, axes = plt.subplots(len(results), 1, figsize=(8, 2.4 * len(results)), sharex=True, sharey=True, layout="constrained")
    for ax, (name, res), color in zip(np.atleast_1d(axes), results.items(), (EST, MEAS, OTHER)):
        m = ~np.isnan(res.innovation)
        z = res.innovation[m] / np.sqrt(res.S[m])
        ax.axhspan(-2, 2, color="#999999", alpha=0.12, linewidth=0, label=r"$\pm 2$")
        ax.plot(sim.t[m], z, "o", color=color, markersize=3, label=r"$e_k/\sqrt{S_k}$")
        ax.axhline(0.0, color=TRUTH, linewidth=0.8)
        ax.set_title(f"{name}   (mean NIS = {np.mean(z**2):.2f})", fontsize=10, loc="left")
        _style(ax)
        ax.legend(loc="upper right", fontsize=8, frameon=False, ncols=2)
    np.atleast_1d(axes)[-1].set_xlabel("time [s]")
    if title:
        fig.suptitle(title)
    return fig
