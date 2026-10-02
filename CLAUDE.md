# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A personal study repo for Kalman filter variants (1D position/velocity, 3D position/velocity/acceleration, and a full 2D IMU/GNSS extended Kalman filter). It has no single entry point — it's a collection of independent scripts and one proper package, at different levels of polish.

## Commands

This project uses `uv` (see `uv.lock`, `pyproject.toml`, `.python-version` = 3.13). Tests use `pytest` and linting uses `ruff` (both dev dependencies, see `[dependency-groups]` in `pyproject.toml`); `[tool.pytest.ini_options]` sets `pythonpath = ["."]` so both `imu_gnss_ekf_training` and `linear_kf` import as packages regardless of invocation directory.

```bash
# run everything
uv run pytest

# run only one app's suite (they're controlled independently)
uv run pytest tests/imu_gnss_ekf_training
uv run pytest tests/linear_kf

# run a single test
uv run pytest tests/imu_gnss_ekf_training/test_ekf.py::test_ekf_runs_and_tracks_state_reasonably

# run the IMU/GNSS EKF demo (writes outputs/imu_gnss_ekf_training/ekf_summary.png)
uv run python -m imu_gnss_ekf_training.demo
uv run python -m imu_gnss_ekf_training.demo --total-time 90 --gnss-dropout 0.35 --output-dir outputs/demo_run

# run the GNSS-dropout comparison experiment (writes outputs/imu_gnss_ekf_training/dropout_experiments.png)
uv run python -m imu_gnss_ekf_training.experiments

# lint (no formatter/type-checker configured — ruff is lint-only here)
uv run ruff check .
```

`[tool.ruff]` sets `line-length = 145` (chosen to fit the existing NumPy/matrix-heavy lines without reformatting them) and `[tool.ruff.lint]` selects `E, F, I, UP, W` (pycodestyle, pyflakes, isort, pyupgrade) — deliberately not the stricter pydocstyle/pylint/bugbear rule families, which would be noisy for this repo's scratch scripts.

CI (`.github/workflows/tests.yml`) runs two independent jobs on push/PR to `main`: `lint` (`uv run ruff check .`) and `pytest` (`uv run pytest -v`).

## Architecture

### `imu_gnss_ekf_training/` — the one real package (NumPy-only, no pykalman)

A planar (2D) EKF fusing IMU (accelerometer + gyro) and GNSS (position + velocity), with 8-state estimation: `[x, y, vx, vy, yaw, accel_bias_x, accel_bias_y, gyro_bias]` (see `IDX_*` constants in `ekf.py`). Full model derivation is in `imu_gnss_ekf_training/README.md`.

- `simulator.py` — generates ground-truth trajectory + noisy IMU/GNSS measurements, with configurable GNSS dropout probability. `initial_position`/`initial_yaw` set the true start pose (scenarios describe motion relative to it; body-frame IMU/VO data are identical for any start pose). Trajectories are selected by `SimulatorConfig.scenario` (`SCENARIOS`: `demo1`, `demo2`, `stationary`, `forward_back`, `circle`), each a `truth_inputs_*` function of body-frame accel + yaw rate. Keep piecewise-constant phase boundaries on round times so they fall on the sample grid, or discretized accel/decel won't cancel (see `truth_inputs_forward_back`). It also simulates visual odometry (`vo_*` config fields, `simulate_visual_odometry`): noisy SE(2) relative poses between tracked frames, drawn from the RNG after IMU/GNSS so existing seeds are unchanged (see `doc/visual_odometry.md`). Truth propagation duplicates the same kinematics as `ekf.py`'s `predict`, so changes to the process model generally need to be mirrored in both places.
- `ekf.py` — `ImuGnssEkf.predict`/`.update` (Jacobian-based EKF steps) plus `run_filter`, which drives predict/update over a full sequence given `gnss_available` flags and returns posterior plus prior state/covariance and GNSS innovations/`S` (NaN when no update) per step. Covariance is symmetrized (`0.5 * (P + P.T)`) after every predict/update to counter numerical drift. Yaw is wrapped to `[-pi, pi]` via `wrap_angle` after every predict/update. Visual odometry is fused by stochastic cloning: `predict`/`update` also accept the 11-element augmented state (8-state + pose clone at `IDX_CLONE_*`), with `clone_pose`/`update_vo`/`drop_clone`; `run_filter` fuses VO when given the `vo_*` arrays and keeps its returned histories 8-state. Absolute yaw is unobservable with VO but no GNSS, and the estimate-linearized EKF becomes overconfident there (no FEJ yet; see the `xfail` test and `doc/visual_odometry.md` §7).
- `plotting.py` — matplotlib figure builders consumed by `demo.py`/`experiments.py`; doesn't run the filter itself.
- `demo.py` — CLI: simulate → filter → plot → print RMSE metrics (`summarize_errors`). VO is simulated always but fused only with `--use-vo`; `--vo-*` flags set the VO simulation. `--initial-pose X Y YAW_DEG` sets the true world start pose (the EKF always starts at 0, 0, 0); `--no-gnss-update` keeps GNSS in the simulation but not in the filter, anchors the EKF's initial pose prior tightly (the nav frame is the start frame), and writes `ekf_frame_alignment.png` + alignment metrics. Argument parsing is duplicated (not shared) with `experiments.py`.
- `experiments.py` — CLI: runs `demo`'s pipeline twice (dense vs. intermittent GNSS) and produces a comparison figure.
- `navlog.py` — `NavLog` dataclass: everything to replay one run (posterior + prior state/covariance, GNSS rows NaN where absent, `R`, innovations/`S`, optional truth/IMU, `metadata_json` with `schema_version` and `EkfConfig`), saved/loaded as `.npz`. `demo.py`/`run_ekf_from_csv.py` write it via `--save-log PATH`. Bump `SCHEMA_VERSION` when changing the schema.
- `alignment.py` — SE(2) trajectory alignment (`fit_rigid_2d`, `align_trajectory`): compares a start-anchored nav-frame estimate with world-frame truth, via the true start pose and via a least-squares best fit (aligned ATE).
- `misc/analyze_vo_only.py` — regenerates `figures/vo_only_*.png` and the tables in `doc/vo_only_navigation.md` (IMU + VO without GNSS: reference run, VO noise sweep, gyro-bias experiment; ~4 min). Rerun it and update the doc's numbers after changing the filter, VO model or scenarios.
- `ellipse.py` — pure-NumPy confidence-ellipse math (`covariance_ellipse`, chi-square(2) scale, Mahalanobis).
- `animation.py` / `animate.py` — `NavAnimator` (2D trajectory, `--view trajectory`) and `BiasAnimator` (bias estimate vs. truth per bias, `--view bias`) render a `NavLog` as mp4 (ffmpeg) or gif via the shared `_Movie` base; artists are built once and mutated per frame, and constrained layout is solved once then frozen (re-solving per frame was the dominant render cost).
- `misc/analyze_bias_convergence.py` — regenerates `figures/bias_convergence_*.png` and the tables in `doc/bias_convergence.md` (bias error vs. ±2σ, Kalman gain rows recomputed from the log as `P_prior H^T S^-1`, and a truth-linearized covariance reference that calls `ImuGnssEkf.predict/update` at the true state with noise-free IMU input). Rerun it and update the doc's numbers after changing the filter or scenarios.
- Exports for external use come from `imu_gnss_ekf_training/__init__.py` (`EkfConfig`, `ImuGnssEkf`, `run_filter`, `SimulatorConfig`, `simulate_scenario`); `tests/` imports from there rather than reaching into submodules directly (except for `IDX_*` constants and `ImuGnssEkf`, which are imported from `imu_gnss_ekf_training.ekf`).
- `EkfConfig`/`SimulatorConfig` intentionally duplicate several noise parameters (accel/gyro noise, bias walk stds, GNSS stds). `SimulatorConfig` specifies IMU white noise as a density (`accel_noise_density`, `gyro_noise_density`) with per-sample `accel_noise_std`/`gyro_noise_std` derived as properties (`density / sqrt(dt)`), while `EkfConfig` takes the per-sample std directly — callers (`demo.py`, `experiments.py`, tests) construct an `EkfConfig` from a `SimulatorConfig`'s values by hand; there's no shared conversion helper.

### `linear_kf/` — linear KF study scripts, organized into subpackages

Layout (run instructions, incl. why the working directory matters, are in `linear_kf/README.md`):

- `linear_kf/simple/` — from-scratch NumPy implementations: `kf_pv.py` (1D position/velocity) and `kf_pva3d.py` (`KalmanFilterPVA_RandomAcc3d`, 3D position/velocity/acceleration).
- `linear_kf/pykalman_check/` — the same models built on `pykalman`, for cross-checking: `kf_pv_pykalman.py`, `kf_pv_em_pykalman.py`, `kf_pva3d_pykalman.py` (imports the class from `simple/kf_pva3d.py`).
- `linear_kf/process_noise_sim/` — process-noise Monte Carlo simulations (`process_noise.py` random acceleration, `process_noise2.py` random jerk) and their derivation in `doc/process_noise.md`.
- `linear_kf/imu_1d/` — import-safe 1D KF with the accelerometer as control input (`x = [p, v]`, `F`/`B` zero-order hold, `Q = sigma_d^2 B B^T`) and position updates: `simulator.py` (truth integrated with the same ZOH step as the filter, so the motion model is exact; optional unmodelled `accel_bias` and `pos_outage`), `kf.py` (`KalmanFilterImu1d`, `run_filter`, position-only baseline `run_position_only_filter`), `demo.py` CLI (`--output-dir`), `make_doc_figures.py` (regenerates `imu_1d/doc/*.png` and prints the Monte Carlo table in `doc/kf_1d_imu.md`; rerun and update the doc's numbers after changing the simulator, filter or defaults). Tested in `tests/linear_kf/test_imu_1d.py`.
- Shared helpers stay at the `linear_kf/` top level: `x_generator.py` (truth trajectory generators), `kf_pv_plot.py`, `kf_pva_plot.py`.

Internal imports are relative (`from ..x_generator import ...`), so scripts run via `-m` from the repo root (e.g. `uv run python -m linear_kf.pykalman_check.kf_pv_pykalman`), not as bare `python linear_kf/pykalman_check/kf_pv_pykalman.py`. Most of these files (`simple/kf_pv.py`, `pykalman_check/kf_pv_em_pykalman.py`, `pykalman_check/kf_pv_pykalman.py`, `process_noise_sim/process_noise.py`, `process_noise_sim/process_noise2.py`) are exploratory scripts that simulate, plot, and `savefig(...)` at module scope with no `if __name__ == "__main__":` guard — running or importing them has side effects (writes PNGs into the current directory). `x_generator.py` (trajectory generators) and `simple/kf_pva3d.py`'s `KalmanFilterPVA_RandomAcc3d` class (its own `__main__` block is guarded) are the only import-safe, unit-testable pieces, and are what `tests/linear_kf/` covers.

### Root-level files

`main.py` and `sample_kf_pva3d.json` are leftover/placeholder artifacts, not part of any package. `README.md` is currently empty — `imu_gnss_ekf_training/README.md` is the real documentation for that package.

### Two independent Kalman-filter implementation styles coexist

- `imu_gnss_ekf_training/ekf.py`: hand-rolled predict/update using `np.linalg.solve` (avoids explicit matrix inversion) and the Joseph-form covariance update.
- `linear_kf/pykalman_check/*_pykalman.py`: same kind of models built on the `pykalman` library's `KalmanFilter.filter`/`.smooth`/`.em`, useful for cross-checking the from-scratch math.

When editing filter math, check whether the corresponding `pykalman`-based script in `linear_kf/pykalman_check/` encodes the same model (F/Q construction) and should be kept consistent.
