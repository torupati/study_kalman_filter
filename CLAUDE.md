# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A personal study repo for Kalman filter variants (1D position/velocity, 3D position/velocity/acceleration, and a full 2D IMU/GNSS extended Kalman filter). It has no single entry point — it's a collection of independent scripts and one proper package, at different levels of polish.

## Commands

This project uses `uv` (see `uv.lock`, `pyproject.toml`, `.python-version` = 3.13). Tests use `pytest` and linting uses `ruff` (both dev dependencies, see `[dependency-groups]` in `pyproject.toml`); `[tool.pytest.ini_options]` sets `pythonpath = ["."]` so both `imu_gnss_ekf_training` and `my_test_kf` import as packages regardless of invocation directory.

```bash
# run everything
uv run pytest

# run only one app's suite (they're controlled independently)
uv run pytest tests/imu_gnss_ekf_training
uv run pytest tests/my_test_kf

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

- `simulator.py` — generates ground-truth trajectory + noisy IMU/GNSS measurements, with configurable GNSS dropout probability. Truth propagation duplicates the same kinematics as `ekf.py`'s `predict`, so changes to the process model generally need to be mirrored in both places.
- `ekf.py` — `ImuGnssEkf.predict`/`.update` (Jacobian-based EKF steps) plus `run_filter`, which drives predict/update over a full sequence given `gnss_available` flags. Covariance is symmetrized (`0.5 * (P + P.T)`) after every predict/update to counter numerical drift. Yaw is wrapped to `[-pi, pi)` after every predict/update via `wrap_angle`.
- `plotting.py` — matplotlib figure builders consumed by `demo.py`/`experiments.py`; doesn't run the filter itself.
- `demo.py` — CLI: simulate → filter → plot → print RMSE metrics (`summarize_errors`). Argument parsing is duplicated (not shared) with `experiments.py`.
- `experiments.py` — CLI: runs `demo`'s pipeline twice (dense vs. intermittent GNSS) and produces a comparison figure.
- Exports for external use come from `imu_gnss_ekf_training/__init__.py` (`EkfConfig`, `ImuGnssEkf`, `run_filter`, `SimulatorConfig`, `simulate_scenario`); `tests/` imports from there rather than reaching into submodules directly (except for `IDX_*` constants and `ImuGnssEkf`, which are imported from `imu_gnss_ekf_training.ekf`).
- `EkfConfig`/`SimulatorConfig` intentionally duplicate several noise parameters (accel/gyro noise, bias walk stds, GNSS stds) — callers (`demo.py`, `experiments.py`, tests) construct an `EkfConfig` from a `SimulatorConfig`'s values by hand; there's no shared conversion helper.

### `my_test_kf/` — standalone scratch scripts, now a real package

Internal imports are relative (`from .x_generator import ...`), so scripts run via `-m` from the repo root (e.g. `uv run python -m my_test_kf.kf_pv_pykalman`), not as bare `python my_test_kf/kf_pv_pykalman.py`. Most of these files (`kf_pv.py`, `kf_pv_em_pykalman.py`, `kf_pv_pykalman.py`, `process_noise.py`, `process_noise2.py`) are exploratory scripts that simulate, plot, and `savefig(...)` at module scope with no `if __name__ == "__main__":` guard — running or importing them has side effects (writes PNGs into the current directory). `x_generator.py` (trajectory generators) and `kf_pva3d.py`'s `KalmanFilterPVA_RandomAcc3d` class (its own `__main__` block is guarded) are the only import-safe, unit-testable pieces, and are what `tests/my_test_kf/` covers.

### Root-level files

`main.py` and `sample_kf_pva3d.json` are leftover/placeholder artifacts, not part of any package. `README.md` is currently empty — `imu_gnss_ekf_training/README.md` is the real documentation for that package.

### Two independent Kalman-filter implementation styles coexist

- `imu_gnss_ekf_training/ekf.py`: hand-rolled predict/update using `np.linalg.solve` (avoids explicit matrix inversion) and the Joseph-form covariance update.
- `my_test_kf/*_pykalman.py`: same kind of models built on the `pykalman` library's `KalmanFilter.filter`/`.smooth`/`.em`, useful for cross-checking the from-scratch math.

When editing filter math, check whether the corresponding `pykalman`-based script in `my_test_kf/` encodes the same model (F/Q construction) and should be kept consistent.
