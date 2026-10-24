"""Regenerate the figures and tables in linear_kf/imu_1d/doc/fgo_1d_imu.md.

    uv run python -m linear_kf.imu_1d.make_fgo_doc_figures

Rerun it and update the doc's numbers after changing the simulator, the filter, the factor graph or the defaults.
"""

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .demo import P0, X0, run_imu_kf
from .fgo import FgoResult, PriorFactor, build_graph, interpolate, run_batch_fgo, run_sliding_window_fgo
from .kf import FilterResult, KalmanFilterImu1d, run_filter
from .plotting import EST, MEAS, OTHER, TRUTH, _shade_outage, _style
from .simulator import SimConfig, SimResult, simulate

DOC_DIR = Path(__file__).parent / "doc"
SEED = 0
N_MONTE_CARLO = 50
WINDOWS = (2, 3, 5)
WINDOW_COLORS = ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")  # ordinal blue ramp: window 2, 3, 5, batch
OUTAGE = (30.0, 50.0)
WRONG_P0 = 20.0  # [m] prior position mean in section 7.1; the truth starts at 0
WRONG_PRIOR_SIGMAS = (30.0, 1.0, 0.1)  # [m] prior position std: honest, overconfident, very overconfident
BIAS = 0.05  # [m/s^2] true accelerometer bias in section 10
BIAS_PRIOR_STD = 0.1  # [m/s^2] prior std of the bias node
BIAS_WINDOW = 5


def save(fig, path: Path):
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"wrote {path}")


def fgo_args(sim: SimResult):
    cfg = sim.config
    return sim.t, sim.acc_meas, sim.pos_meas, cfg.accel_noise_std, cfg.pos_noise_std, X0, P0


def plot_graph_schematic():
    """Keyframe nodes, prior / position / IMU factors, and the IMU samples each IMU factor preintegrates."""
    fig, ax = plt.subplots(figsize=(8, 2.8), layout="constrained")
    xs = np.arange(5) * 2.0
    for i, x in enumerate(xs):
        for k in range(1, 8):  # IMU samples between keyframes
            if i < len(xs) - 1:
                ax.plot(x + 2.0 * k / 8, -0.55, ".", color="#999999", markersize=3)
        if i < len(xs) - 1:
            ax.plot([x, x + 2.0], [0, 0], color=TRUTH, linewidth=1.0, zorder=1)
            ax.plot(x + 1.0, 0, "s", color=EST, markersize=10, zorder=3)
            ax.annotate("", (x + 1.85, -0.55), (x + 0.15, -0.55), arrowprops={"arrowstyle": "->", "color": "#999999", "linewidth": 0.8})
        if i < len(xs) - 1:  # the last node is the last IMU sample, with no fix
            ax.plot([x, x], [0, 0.8], color=TRUTH, linewidth=1.0, zorder=1)
            ax.plot(x, 0.8, "s", color=MEAS, markersize=10, zorder=3)
            ax.text(x + 0.18, 0.8, f"$z_{i}$", va="center", fontsize=10)
        ax.plot(x, 0, "o", color="white", markeredgecolor=TRUTH, markersize=24, markeredgewidth=1.5, zorder=4)
        ax.text(x, 0, f"$x_{i}$", ha="center", va="center", fontsize=11, zorder=5)
    ax.plot([xs[0] - 1.0, xs[0]], [0, 0], color=TRUTH, linewidth=1.0, zorder=1)
    ax.plot(xs[0] - 1.0, 0, "s", color=TRUTH, markersize=10, zorder=3)
    ax.text(xs[0] - 1.0, -0.3, "prior", ha="center", va="top", fontsize=9)
    ax.text(xs[1] + 1.0, 0.25, "IMU factor", ha="center", va="bottom", fontsize=9, color=EST)
    ax.text(xs[1] + 1.0, 1.2, "position factor", ha="center", va="center", fontsize=9, color=MEAS)
    ax.text(xs[2] + 1.0, -0.75, "100 accelerometer samples, preintegrated", ha="center", va="top", fontsize=9, color="#777777")
    ax.text(xs[-1], -0.3, "last sample", ha="center", va="top", fontsize=9)
    ax.set_xlim(xs[0] - 1.6, xs[-1] + 0.8)
    ax.set_ylim(-1.1, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig


def plot_information_structure(sim: SimResult):
    """Sparsity of the whitened Jacobian A and the information matrix H = A^T A."""
    cfg = sim.config
    graph, _ = build_graph(*fgo_args(sim))
    A, _ = graph.linearize(np.zeros((graph.n_nodes, 2)))
    H = A.T @ A
    kinds = {"PriorFactor": TRUTH, "PositionFactor": MEAS, "ImuFactor": EST}
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2), layout="constrained", gridspec_kw={"width_ratios": [1, 1.25]})
    row = 0
    for f in graph.factors:
        r, _ = f.linearize([np.zeros(2) for _ in f.keys])
        rr, cc = np.nonzero(np.abs(A[row : row + len(r)]) > 1e-12)
        axes[0].plot(cc, rr + row, "s", color=kinds[type(f).__name__], markersize=4)
        row += len(r)
    axes[0].set_xlim(-0.5, A.shape[1] - 0.5)
    axes[0].set_ylim(A.shape[0] - 0.5, -0.5)
    axes[0].set_title(f"whitened Jacobian $A$ ({A.shape[0]} x {A.shape[1]})", fontsize=10)
    axes[0].set_xlabel("state column (2 per keyframe)")
    axes[0].set_ylabel("residual row")
    for name, color in kinds.items():
        axes[0].plot([], [], "s", color=color, label=name.removesuffix("Factor").lower())
    axes[0].legend(loc="lower left", fontsize=8, frameon=False)
    rr, cc = np.nonzero(np.abs(H) > 1e-12)
    axes[1].plot(cc, rr, "s", color=EST, markersize=6)
    axes[1].set_xlim(-0.5, H.shape[1] - 0.5)
    axes[1].set_ylim(H.shape[0] - 0.5, -0.5)
    axes[1].set_aspect("equal")
    axes[1].set_title(r"information $H = A^\top A$ (block tridiagonal)", fontsize=10)
    axes[1].set_xlabel("state column")
    for ax in axes:
        _style(ax)
        ax.set_xticks(np.arange(0, A.shape[1], 2))
    fig.suptitle(f"{cfg.duration:.0f} s at {cfg.pos_rate:.0f} Hz position: {graph.n_nodes} keyframes")
    return fig


def plot_kf_vs_batch(sim: SimResult):
    """Error and +-2 sigma of the KF (filter) and batch FGO (smoother) on the same data."""
    kf = run_imu_kf(sim)
    fgo = run_batch_fgo(*fgo_args(sim))
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True, layout="constrained")
    truth = (sim.pos_true, sim.vel_true)
    labels = ("position error [m]", "velocity error [m/s]")
    for i, ax in enumerate(axes):
        for name, res, color in (("KF", kf, MEAS), ("batch FGO", fgo, EST)):
            sig = np.sqrt(res.P[:, i, i])
            ax.fill_between(sim.t, -2 * sig, 2 * sig, color=color, alpha=0.13, linewidth=0)
            ax.plot(sim.t, 2 * sig, color=color, linewidth=0.8, linestyle="--")
            ax.plot(sim.t, -2 * sig, color=color, linewidth=0.8, linestyle="--")
            ax.plot(sim.t, res.x[:, i] - truth[i], color=color, linewidth=1.2, label=f"{name} error, $\\pm 2\\sigma$ dashed")
        ax.axhline(0.0, color=TRUTH, linewidth=0.8)
        ax.set_ylabel(labels[i])
        _style(ax)
        _shade_outage(ax, sim)
        ax.legend(loc="upper left", fontsize=8, frameon=False)
    axes[1].set_ylim(-1.2, 1.2)  # the KF's start-up transient (initial velocity off by 1 m/s) is clipped
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(f"filter vs smoother, no position measurements for {OUTAGE[0]:.0f}-{OUTAGE[1]:.0f} s")
    return fig, kf, fgo


def plot_window_sigma(sim: SimResult):
    """Position std at the keyframes: KF (online), sliding windows (fixed lag) and batch."""
    kf = run_imu_kf(sim)
    fgo = run_batch_fgo(*fgo_args(sim))
    fig, ax = plt.subplots(figsize=(8, 3.6), layout="constrained")
    key_t = sim.t[fgo.key_idx]
    gap = np.flatnonzero(np.diff(key_t) > 1.5) + 1  # no line across the outage: there are no keyframes in it

    def broken(y):
        return np.insert(key_t, gap, np.nan), np.insert(y, gap, np.nan)

    ax.plot(*broken(np.sqrt(kf.P[fgo.key_idx, 0, 0])), "o-", color=MEAS, linewidth=1.2, markersize=4, label="KF (= newest window node)")
    for w, color in zip(WINDOWS, WINDOW_COLORS, strict=False):
        sw = run_sliding_window_fgo(*fgo_args(sim), window=w)
        ax.plot(*broken(np.sqrt(sw.P_key[:, 0, 0])), "o-", color=color, linewidth=1.2, markersize=4, label=f"window {w} (lag {w - 1})")
    ax.plot(*broken(np.sqrt(fgo.solution.cov.diagonal()[0::2])), "o-", color=WINDOW_COLORS[-1], linewidth=1.2, markersize=4, label="batch")
    ax.set_ylim(0, None)
    ax.set_xlim(20, 60)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("position std at keyframes [m]")
    _style(ax)
    _shade_outage(ax, sim)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle("each keyframe's final estimate: more lag, more future fixes, smaller std")
    return fig


def run_with_prior(sim: SimResult, p0: float, sigma_p0: float) -> tuple[FilterResult, FgoResult]:
    """KF and batch FGO from the prior N([p0, 0], diag(sigma_p0^2, 2^2)); the velocity prior is the default one."""
    x0, p0_cov = np.array([p0, X0[1]]), np.diag([sigma_p0**2, P0[1, 1]])
    cfg = sim.config
    kf = run_filter(KalmanFilterImu1d(cfg.accel_noise_std, cfg.pos_noise_std), sim.t, sim.acc_meas, sim.pos_meas, x0, p0_cov)
    return kf, run_batch_fgo(sim.t, sim.acc_meas, sim.pos_meas, cfg.accel_noise_std, cfg.pos_noise_std, x0, p0_cov)


def run_fgo_without_prior(sim: SimResult) -> FgoResult:
    """Batch FGO with the prior factor removed: the fixes alone make every keyframe observable."""
    cfg = sim.config
    graph, key_idx = build_graph(*fgo_args(sim))
    graph.factors = [f for f in graph.factors if not isinstance(f, PriorFactor)]
    sol = graph.solve()
    x, P = interpolate(sim.t, sim.acc_meas, key_idx, sol, cfg.accel_noise_std)
    return FgoResult(x, P, key_idx, sol)


def plot_wrong_prior(sim: SimResult, sigma_p0: float):
    """Error and +-2 sigma when the prior position is WRONG_P0 off with std sigma_p0: KF, FGO, FGO without the prior."""
    kf, fgo = run_with_prior(sim, WRONG_P0, sigma_p0)
    free = run_fgo_without_prior(sim)
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True, layout="constrained")
    truth = (sim.pos_true, sim.vel_true)
    labels = ("position error [m]", "velocity error [m/s]")
    m = sim.t <= 30.0
    for i, ax in enumerate(axes):
        for name, res, color in (("KF", kf, MEAS), ("batch FGO", fgo, EST), ("batch FGO, no prior factor", free, OTHER)):
            sig = np.sqrt(res.P[m, i, i])
            ax.plot(sim.t[m], 2 * sig, color=color, linewidth=0.8, linestyle="--")
            ax.plot(sim.t[m], -2 * sig, color=color, linewidth=0.8, linestyle="--")
            ax.plot(sim.t[m], res.x[m, i] - truth[i][m], color=color, linewidth=1.4, label=f"{name} error, $\\pm 2\\sigma$ dashed")
        ax.axhline(0.0, color=TRUTH, linewidth=0.8)
        ax.set_ylabel(labels[i])
        _style(ax)
    axes[0].legend(loc="upper right", fontsize=8, frameon=False)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(f"prior position {WRONG_P0:.0f} m off (truth 0 m) with $\\sigma$ = {sigma_p0:g} m: confidently wrong")
    return fig


def print_wrong_prior_numbers(sim: SimResult):
    """Section 7.1: errors after a wrong prior, how long it lasts, the first-update NIS / chi2, and FGO without a prior.

    "Influence gone" is the first time after which the estimate stays within 0.1 m of the run with the honest
    (widest) prior, i.e. when the wrong prior no longer matters.
    """
    ts = (0.0, 1.0, 5.0, 10.0, 20.0, 30.0)
    idx = [int(round(t * sim.config.imu_rate)) for t in ts]
    print(f"\nwrong prior p0 = {WRONG_P0:.0f} m: position error at t = {ts}")
    honest = run_with_prior(sim, WRONG_P0, WRONG_PRIOR_SIGMAS[0])
    for sigma_p0 in WRONG_PRIOR_SIGMAS:
        kf, fgo = run_with_prior(sim, WRONG_P0, sigma_p0)
        for name, res, ref in (("KF", kf, honest[0]), ("FGO", fgo, honest[1])):
            e = res.x[:, 0] - sim.pos_true
            z = np.abs(e) / np.sqrt(res.P[:, 0, 0])
            off = np.flatnonzero(np.abs(res.x[:, 0] - ref.x[:, 0]) >= 0.1)
            gone = sim.t[off[-1] + 1] if len(off) else 0.0
            extra = f"first NIS {kf.nis[0]:.2f}" if name == "KF" else f"chi2 {fgo.solution.chi2:.0f} / dof {fgo.solution.dof}"
            print(
                f"  sigma_p0 = {sigma_p0:>4g} {name:3s} {np.round(e[idx], 2)}  max|vel err| {np.max(np.abs(res.x[:, 1] - sim.vel_true)):5.2f}"
                f"  max |err|/sigma {z.max():6.1f}  influence gone at {gone:5.2f} s  {extra}"
            )
    free = run_fgo_without_prior(sim)
    print(
        f"  FGO without prior: node 0 pos err {free.x[0, 0] - sim.pos_true[0]:.3f} m, vel err {free.x[0, 1] - sim.vel_true[0]:.3f} m/s,"
        f" sigma {np.sqrt(free.P[0, 0, 0]):.3f} m / {np.sqrt(free.P[0, 1, 1]):.3f} m/s, chi2 {free.solution.chi2:.1f} / dof {free.solution.dof},"
        f" max |pos diff| vs honest-prior FGO {np.max(np.abs(free.x[:, 0] - honest[1].x[:, 0])):.3f} m"
    )


def bias_runs(sim: SimResult):
    """KF without a bias state, batch FGO without and with the bias node, and the sliding window with it."""
    return (
        run_imu_kf(sim),
        run_batch_fgo(*fgo_args(sim)),
        run_batch_fgo(*fgo_args(sim), bias_prior_std=BIAS_PRIOR_STD),
        run_sliding_window_fgo(*fgo_args(sim), window=BIAS_WINDOW, bias_prior_std=BIAS_PRIOR_STD),
    )


def plot_bias(sim: SimResult):
    """Bias estimate over time (online and batch) and the velocity error with and without the bias node."""
    kf, plain, fgo, sw = bias_runs(sim)
    tk = sim.t[sw.key_idx]
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True, layout="constrained")
    ax = axes[0]
    lo, hi = sw.bias_online - 2 * sw.bias_std_online, sw.bias_online + 2 * sw.bias_std_online
    ax.fill_between(tk, lo, hi, color=MEAS, alpha=0.13, linewidth=0)
    online_label = "online estimate, $\\pm 2\\sigma$ (newest window node = augmented KF)"
    ax.plot(tk, sw.bias_online, "o-", color=MEAS, linewidth=1.2, markersize=3, label=online_label)
    ax.axhspan(fgo.bias - 2 * fgo.bias_std, fgo.bias + 2 * fgo.bias_std, color=EST, alpha=0.13, linewidth=0)
    ax.axhline(fgo.bias, color=EST, linewidth=1.4, label=f"batch FGO: {fgo.bias:.4f} $\\pm$ {2 * fgo.bias_std:.4f} (2$\\sigma$)")
    ax.axhline(BIAS, color=TRUTH, linewidth=1.0, linestyle="--", label=f"true bias {BIAS} m/s$^2$")
    ax.set_ylim(-0.15, 0.25)
    ax.set_ylabel("accel bias [m/s$^2$]")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax = axes[1]
    for name, res, color in (("KF, no bias state", kf, MEAS), ("batch FGO, no bias node", plain, OTHER), ("batch FGO with bias node", fgo, EST)):
        ax.plot(sim.t, res.x[:, 1] - sim.vel_true, color=color, linewidth=1.2, label=name)
    sig = np.sqrt(fgo.P[:, 1, 1])
    ax.fill_between(sim.t, -2 * sig, 2 * sig, color=EST, alpha=0.13, linewidth=0, label="$\\pm 2\\sigma$, FGO with bias node")
    ax.axhline(0.0, color=TRUTH, linewidth=0.8)
    ax.set_ylim(-0.4, 0.5)
    ax.set_ylabel("velocity error [m/s]")
    ax.set_xlabel("time [s]")
    ax.legend(loc="lower right", fontsize=8, frameon=False, ncols=2)
    for a in axes:
        _style(a)
    fig.suptitle(f"constant accelerometer bias {BIAS} m/s$^2$, estimated as one node")
    return fig


def print_bias_numbers(sim: SimResult):
    """Section 10: online bias convergence, velocity RMSE at the ends vs the middle, and the noise limit on the bias std."""
    kf, plain, fgo, sw = bias_runs(sim)
    tk = sim.t[sw.key_idx]
    print(f"\nbias {BIAS}: batch {fgo.bias:.4f} +- {fgo.bias_std:.4f}")
    for T in (2, 5, 10, 20, 30, 60):
        i = int(np.flatnonzero(np.isclose(tk, T))[0])
        print(f"  online at t = {T:2d} s: {sw.bias_online[i]:+.4f} +- {sw.bias_std_online[i]:.4f}")
    for lo, hi in ((0.0, 5.0), (5.0, 55.0), (55.0, 60.01)):
        m = (sim.t >= lo) & (sim.t < hi)
        rmse = [np.sqrt(np.mean((r.x[m, 1] - sim.vel_true[m]) ** 2)) for r in (kf, plain, fgo)]
        print(f"  vel RMSE {lo:.0f}-{hi:.0f} s: KF {rmse[0]:.3f}, FGO no node {rmse[1]:.3f}, FGO + node {rmse[2]:.3f}")
    for density in (0.05, 0.005, 0.0005):
        s = simulate(replace(sim.config, accel_noise_density=density), np.random.default_rng(SEED))
        print(f"  accel noise density {density}: batch bias std {run_batch_fgo(*fgo_args(s), bias_prior_std=BIAS_PRIOR_STD).bias_std:.5f}")


def bias_monte_carlo_table(cfg: SimConfig):
    print(f"\n(bias {BIAS} m/s^2, {N_MONTE_CARLO} seeds, t >= 5 s)")
    print("| estimator | pos RMSE [m] | pos sigma [m] | vel RMSE [m/s] | vel sigma [m/s] | chi2/dof |")
    print("|---|---|---|---|---|---|")
    rows: dict[str, list[dict[str, float]]] = {}
    z, online_kf, online_aug = [], [], []
    for seed in range(N_MONTE_CARLO):
        sim = simulate(cfg, np.random.default_rng(seed))
        kf, plain, fgo, sw = bias_runs(sim)
        k = sw.key_idx[sim.t[sw.key_idx] >= 20.0]  # online velocity at keyframes, after the bias has settled
        online_kf.append(kf.x[k, 1] - sim.vel_true[k])
        online_aug.append(sw.x_online[sim.t[sw.key_idx] >= 20.0, 1] - sim.vel_true[k])
        for name, res, chi2 in (
            ("KF, no bias state", kf, np.nan),
            ("batch FGO, no bias node", plain, plain.solution.chi2 / plain.solution.dof),
            (f"window {BIAS_WINDOW} + bias node", sw, np.nan),
            ("batch FGO + bias node", fgo, fgo.solution.chi2 / fgo.solution.dof),
        ):
            rows.setdefault(name, []).append({**summarize(sim, res.x, res.P), "chi2": chi2})
        z.append((fgo.bias - BIAS) / fgo.bias_std)
    for name, rs in rows.items():
        mean = {k: np.mean([r[k] for r in rs]) for k in rs[0]}
        chi2 = "" if np.isnan(mean["chi2"]) else f"{mean['chi2']:.2f}"
        print(f"| {name} | {mean['pos_rmse']:.3f} | {mean['pos_sigma']:.3f} | {mean['vel_rmse']:.3f} | {mean['vel_sigma']:.3f} | {chi2} |")
    print(f"  batch bias: mean (b - b_true) / sigma = {np.mean(z):+.2f}, rms = {np.sqrt(np.mean(np.square(z))):.2f}")
    for name, e in (("2-state KF", np.concatenate(online_kf)), ("augmented KF (newest window node)", np.concatenate(online_aug))):
        print(f"  online velocity error at keyframes, t >= 20 s, {name}: mean {e.mean():+.3f}, rms {np.sqrt(np.mean(e**2)):.3f}")


def summarize(sim: SimResult, x: np.ndarray, P: np.ndarray, skip: float = 5.0) -> dict[str, float]:
    m = sim.t >= skip
    return {
        "pos_rmse": float(np.sqrt(np.mean((x[m, 0] - sim.pos_true[m]) ** 2))),
        "pos_sigma": float(np.sqrt(np.mean(P[m, 0, 0]))),
        "vel_rmse": float(np.sqrt(np.mean((x[m, 1] - sim.vel_true[m]) ** 2))),
        "vel_sigma": float(np.sqrt(np.mean(P[m, 1, 1]))),
    }


def monte_carlo_table(cases: list[tuple[str, SimConfig]]):
    print(f"\n| case | estimator | pos RMSE [m] | pos sigma [m] | vel RMSE [m/s] | vel sigma [m/s] |  ({N_MONTE_CARLO} seeds, t >= 5 s)")
    print("|---|---|---|---|---|---|")
    for name, cfg in cases:
        rows: dict[str, list[dict[str, float]]] = {}
        chi2_dof, nees = [], []
        for seed in range(N_MONTE_CARLO):
            sim = simulate(cfg, np.random.default_rng(seed))
            kf = run_imu_kf(sim)
            rows.setdefault("KF", []).append(summarize(sim, kf.x, kf.P))
            for w in WINDOWS:
                sw = run_sliding_window_fgo(*fgo_args(sim), window=w)
                rows.setdefault(f"window {w}", []).append(summarize(sim, sw.x, sw.P))
            fgo = run_batch_fgo(*fgo_args(sim))
            rows.setdefault("batch FGO", []).append(summarize(sim, fgo.x, fgo.P))
            chi2_dof.append(fgo.solution.chi2 / fgo.solution.dof)
            e = np.stack([fgo.x[:, 0] - sim.pos_true, fgo.x[:, 1] - sim.vel_true], axis=1)
            nees.append(np.mean(np.einsum("na,nab,nb->n", e, np.linalg.inv(fgo.P), e)))
        for est, rs in rows.items():
            mean = {k: np.mean([r[k] for r in rs]) for k in rs[0]}
            print(f"| {name} | {est} | {mean['pos_rmse']:.3f} | {mean['pos_sigma']:.3f} | {mean['vel_rmse']:.3f} | {mean['vel_sigma']:.3f} |")
        print(f"  batch FGO: mean chi2/dof = {np.mean(chi2_dof):.3f}, mean NEES = {np.mean(nees):.3f} (expect 1 and 2)")


def print_sigma_numbers(sim: SimResult, sim_outage: SimResult, kf_o, fgo_o):
    """Steady-state std mid-run (fix at t = 30 s, halfway, just before the next fix), the run ends, and the outage peak."""
    kf, fgo = run_imu_kf(sim), run_batch_fgo(*fgo_args(sim))
    i_fix = int(round(30.0 * sim.config.imu_rate))
    i_mid, i_before = i_fix + 50, i_fix + 99
    for name, res in (("KF", kf), ("FGO", fgo)):
        s, v = np.sqrt(res.P[:, 0, 0]), np.sqrt(res.P[:, 1, 1])
        print(
            f"{name}: pos std at fix {s[i_fix]:.4f}, mid {s[i_mid]:.4f}, before next fix {s[i_before]:.4f}, "
            f"at t=0 {s[0]:.4f}, at end {s[-1]:.4f}; vel std at fix {v[i_fix]:.4f}, at t=0 {v[0]:.4f}, at end {v[-1]:.4f}"
        )
    m = (sim_outage.t >= OUTAGE[0]) & (sim_outage.t < OUTAGE[1])
    for name, res in (("KF", kf_o), ("FGO", fgo_o)):
        s = np.sqrt(res.P[:, 0, 0])
        k = np.flatnonzero(m)[np.argmax(s[m])]
        print(f"{name}: outage peak pos std {s[k]:.3f} at t = {sim_outage.t[k]:.2f} s, vel std there {np.sqrt(res.P[k, 1, 1]):.3f}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DOC_DIR)
    p.add_argument("--skip-monte-carlo", action="store_true")
    args = p.parse_args(argv)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    base = SimConfig()
    outage = replace(base, pos_outage=OUTAGE)

    save(plot_graph_schematic(), out / "fgo_graph.png")
    save(plot_information_structure(simulate(replace(base, duration=8.0), np.random.default_rng(SEED))), out / "fgo_information.png")
    sim_o = simulate(outage, np.random.default_rng(SEED))
    fig, kf_o, fgo_o = plot_kf_vs_batch(sim_o)
    save(fig, out / "fgo_kf_vs_batch_outage.png")
    save(plot_window_sigma(sim_o), out / "fgo_window_sigma.png")

    sim = simulate(base, np.random.default_rng(SEED))
    save(plot_wrong_prior(sim, WRONG_PRIOR_SIGMAS[-1]), out / "fgo_wrong_prior.png")

    print_sigma_numbers(sim, sim_o, kf_o, fgo_o)
    print_wrong_prior_numbers(sim)

    sim_b = simulate(replace(base, accel_bias=BIAS), np.random.default_rng(SEED))
    save(plot_bias(sim_b), out / "fgo_bias.png")
    print_bias_numbers(sim_b)
    if not args.skip_monte_carlo:
        monte_carlo_table([("sine", base), ("sine, outage 30-50 s", outage)])
        bias_monte_carlo_table(replace(base, accel_bias=BIAS))


if __name__ == "__main__":
    main()
