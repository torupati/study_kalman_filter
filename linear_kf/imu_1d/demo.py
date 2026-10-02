"""Simulate -> filter -> plot -> print metrics for the 1D IMU + position KF.

    uv run python -m linear_kf.imu_1d.demo
    uv run python -m linear_kf.imu_1d.demo --scenario stop_and_go
    uv run python -m linear_kf.imu_1d.demo --outage 30 50 --accel-bias 0.05 --output-dir outputs/imu_1d_bias
"""

import argparse
from pathlib import Path

import numpy as np

from .kf import FilterResult, KalmanFilterImu1d, run_filter, run_position_only_filter
from .plotting import plot_compare_velocity, plot_errors, plot_overview
from .simulator import SCENARIOS, SimConfig, SimResult, simulate

X0 = np.array([0.0, 0.0])  # the filter does not know the true initial velocity
P0 = np.diag([1.0**2, 2.0**2])
POSITION_ONLY_ACCEL_PSD = 1.0  # [m/s^2/sqrt(Hz)], tuned for this trajectory (doc/kf_1d_imu.md section 6)


def run_imu_kf(sim: SimResult) -> FilterResult:
    cfg = sim.config
    kf = KalmanFilterImu1d(cfg.accel_noise_std, cfg.pos_noise_std)
    return run_filter(kf, sim.t, sim.acc_meas, sim.pos_meas, X0, P0)


def run_position_only_kf(sim: SimResult) -> FilterResult:
    return run_position_only_filter(sim.t, sim.pos_meas, POSITION_ONLY_ACCEL_PSD, sim.config.pos_noise_std, X0, P0)


def summarize(sim: SimResult, res: FilterResult, skip: float = 5.0) -> dict[str, float]:
    """RMSE and mean NIS after the first `skip` seconds (initial convergence excluded)."""
    m = sim.t >= skip
    return {
        "pos_rmse": float(np.sqrt(np.mean((res.x[m, 0] - sim.pos_true[m]) ** 2))),
        "vel_rmse": float(np.sqrt(np.mean((res.x[m, 1] - sim.vel_true[m]) ** 2))),
        "pos_sigma_rms": float(np.sqrt(np.mean(res.P[m, 0, 0]))),
        "vel_sigma_rms": float(np.sqrt(np.mean(res.P[m, 1, 1]))),
        "mean_nis": float(np.nanmean(res.nis[m])),
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = SimConfig()
    p.add_argument("--scenario", choices=sorted(SCENARIOS), default=d.scenario, help="truth trajectory")
    p.add_argument("--duration", type=float, default=d.duration)
    p.add_argument("--imu-rate", type=float, default=d.imu_rate)
    p.add_argument("--pos-rate", type=float, default=d.pos_rate)
    p.add_argument("--accel-noise-density", type=float, default=d.accel_noise_density, help="m/s^2/sqrt(Hz)")
    p.add_argument("--accel-bias", type=float, default=d.accel_bias, help="m/s^2, not modelled by the KF")
    p.add_argument("--pos-noise-std", type=float, default=d.pos_noise_std, help="m")
    p.add_argument("--outage", type=float, nargs=2, metavar=("START", "END"), default=None, help="no position measurements in [START, END) s")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/linear_kf/imu_1d"))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = SimConfig(
        scenario=args.scenario,
        duration=args.duration,
        imu_rate=args.imu_rate,
        pos_rate=args.pos_rate,
        accel_noise_density=args.accel_noise_density,
        accel_bias=args.accel_bias,
        pos_noise_std=args.pos_noise_std,
        pos_outage=tuple(args.outage) if args.outage else None,
    )
    sim = simulate(cfg, np.random.default_rng(args.seed))
    res = run_imu_kf(sim)
    res_pos = run_position_only_kf(sim)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    plot_overview(sim, f"scenario {cfg.scenario}: truth and measurements").savefig(out / "imu1d_overview.png", dpi=120)
    plot_errors(sim, res, "IMU + position KF: error and $\\pm 2\\sigma$").savefig(out / "imu1d_errors.png", dpi=120)
    plot_compare_velocity(
        sim, {"IMU + position KF": res, "position-only KF (random acceleration model)": res_pos}, "velocity error"
    ).savefig(out / "imu1d_vs_position_only.png", dpi=120)

    for name, r in (("IMU + position KF", res), ("position-only KF", res_pos)):
        s = summarize(sim, r)
        print(
            f"{name:18s} pos RMSE {s['pos_rmse']:.3f} m (sigma {s['pos_sigma_rms']:.3f})  "
            f"vel RMSE {s['vel_rmse']:.3f} m/s (sigma {s['vel_sigma_rms']:.3f})  mean NIS {s['mean_nis']:.2f}"
        )
    print(f"figures written to {out}/")


if __name__ == "__main__":
    main()
