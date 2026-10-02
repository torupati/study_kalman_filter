"""Regenerate the figures and tables in linear_kf/imu_1d/doc/kf_1d_imu.md.

    uv run python -m linear_kf.imu_1d.make_doc_figures

Rerun it and update the doc's numbers after changing the simulator, the filter or the defaults.
"""

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .demo import run_imu_kf, run_position_only_kf, summarize
from .plotting import plot_compare_velocity, plot_errors, plot_innovation, plot_overview, plot_sigma_zoom
from .simulator import SimConfig, simulate

DOC_DIR = Path(__file__).parent / "doc"
SEED = 0
N_MONTE_CARLO = 50


def save(fig, path: Path):
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"wrote {path}")


def monte_carlo_table(cases: list[tuple[str, SimConfig, str]]):
    print(f"\n| case | filter | pos RMSE [m] | pos sigma [m] | vel RMSE [m/s] | vel sigma [m/s] | mean NIS |  ({N_MONTE_CARLO} seeds, t >= 5 s)")
    print("|---|---|---|---|---|---|---|")
    for name, cfg, which in cases:
        runner = run_imu_kf if which == "imu" else run_position_only_kf
        rows = [summarize(sim, runner(sim)) for sim in (simulate(cfg, np.random.default_rng(s)) for s in range(N_MONTE_CARLO))]
        mean = {k: np.mean([r[k] for r in rows]) for k in rows[0]}
        print(
            f"| {name} | {which} | {mean['pos_rmse']:.3f} | {mean['pos_sigma_rms']:.3f} | "
            f"{mean['vel_rmse']:.3f} | {mean['vel_sigma_rms']:.3f} | {mean['mean_nis']:.2f} |"
        )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DOC_DIR)
    out = p.parse_args(argv).output_dir
    out.mkdir(parents=True, exist_ok=True)

    base = SimConfig()
    outage = replace(base, pos_outage=(30.0, 50.0))
    biased = replace(base, accel_bias=0.05)

    sim = simulate(base, np.random.default_rng(SEED))
    res = run_imu_kf(sim)
    save(plot_overview(sim, "truth and measurements"), out / "imu1d_overview.png")
    save(plot_errors(sim, res, r"IMU + position KF: error and $\pm 2\sigma$"), out / "imu1d_errors.png")
    save(plot_sigma_zoom(sim, res, (10.0, 13.0), "filter std between position updates"), out / "imu1d_sigma_zoom.png")
    save(
        plot_compare_velocity(sim, {"IMU + position KF": res, "position-only KF (random acceleration model)": run_position_only_kf(sim)}),
        out / "imu1d_vs_position_only.png",
    )

    sim_o = simulate(outage, np.random.default_rng(SEED))
    save(plot_errors(sim_o, run_imu_kf(sim_o), "no position measurements for 30-50 s (dead reckoning)"), out / "imu1d_outage.png")

    sim_b = simulate(biased, np.random.default_rng(SEED))
    res_b = run_imu_kf(sim_b)
    save(plot_errors(sim_b, res_b, "accelerometer bias 0.05 m/s$^2$, not modelled"), out / "imu1d_bias_errors.png")
    save(plot_innovation(sim, {"no bias": res, "bias 0.05 m/s$^2$": res_b}, "normalized innovation"), out / "imu1d_innovation.png")

    sim_s = simulate(replace(base, scenario="stop_and_go"), np.random.default_rng(SEED))
    save(plot_overview(sim_s, "scenario stop_and_go: truth and measurements"), out / "imu1d_overview_stop_and_go.png")
    save(
        plot_compare_velocity(
            sim_s, {"IMU + position KF": run_imu_kf(sim_s), "position-only KF (random acceleration model)": run_position_only_kf(sim_s)}
        ),
        out / "imu1d_vs_position_only_stop_and_go.png",
    )

    monte_carlo_table(
        [
            ("sine", replace(base, scenario="sine"), "imu"),
            ("sine", replace(base, scenario="sine"), "position-only"),
            ("stationary", replace(base, scenario="stationary"), "imu"),
            ("stationary", replace(base, scenario="stationary"), "position-only"),
            ("stop_and_go", replace(base, scenario="stop_and_go"), "imu"),
            ("stop_and_go", replace(base, scenario="stop_and_go"), "position-only"),
            ("sine, outage 30-50 s", outage, "imu"),
            ("sine, bias 0.05 m/s^2", biased, "imu"),
            ("stationary, bias 0.05 m/s^2", replace(biased, scenario="stationary"), "imu"),
        ]
    )


if __name__ == "__main__":
    main()
