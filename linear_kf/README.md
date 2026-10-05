# linear_kf

Linear Kalman filter study scripts.

| Directory | Contents |
|---|---|
| `simple/` | From-scratch NumPy Kalman filter / RTS smoother: `kf_pv.py` (1D position/velocity), `kf_pva3d.py` (3D position/velocity/acceleration, `KalmanFilterPVA_RandomAcc3d`) |
| `pykalman_check/` | The same models built on `pykalman`, for cross-checking: `kf_pv_pykalman.py`, `kf_pv_em_pykalman.py` (EM parameter estimation), `kf_pva3d_pykalman.py` |
| `process_noise_sim/` | Monte Carlo checks of process-noise models: `process_noise.py` (random acceleration), `process_noise2.py` (random jerk), `animate_process_noise.py` (movie of the random acceleration model's p/v/a distributions spreading over time). Derivation: [`doc/process_noise.md`](process_noise_sim/doc/process_noise.md) |
| `bayes_1d/` | Bayes estimation of a 1D Gaussian position (prior × likelihood → posterior), and how it equals the KF measurement update: `bayes_gaussian_1d.py`. Explanation: [`doc/bayes_gaussian_1d.md`](bayes_1d/doc/bayes_gaussian_1d.md) |
| `imu_1d/` | 1D Kalman filter with the accelerometer as input and 1 Hz position updates: simulator, filter, demo CLI, tests. Explanation: [`doc/kf_1d_imu.md`](imu_1d/doc/kf_1d_imu.md) |
| `doc/` | Explanatory documents: [`kf_basics.md`](doc/kf_basics.md) (KF/RTS derivation on the 1D position/velocity model) |
| top level | Shared helpers: `x_generator.py` (truth trajectories), `kf_pv_plot.py`, `kf_pva_plot.py` |

## How to run

**The path matters.** The scripts use relative imports (`from ..x_generator import ...`), so run them
as modules with `-m` from the repository root. Running a file directly fails with
`ImportError: attempted relative import with no known parent package`.

```bash
cd <repo root>

# simple/ — from-scratch implementation
uv run python -m linear_kf.simple.kf_pv
uv run python -m linear_kf.simple.kf_pva3d

# pykalman_check/ — same models via pykalman
uv run python -m linear_kf.pykalman_check.kf_pv_pykalman
uv run python -m linear_kf.pykalman_check.kf_pv_em_pykalman

# process_noise_sim/
uv run python -m linear_kf.process_noise_sim.process_noise
uv run python -m linear_kf.process_noise_sim.process_noise2
uv run python -m linear_kf.process_noise_sim.animate_process_noise                 # -> process_noise_anim.mp4 (~2 min to render)
uv run python -m linear_kf.process_noise_sim.animate_process_noise --output pn.gif  # .gif if ffmpeg is unavailable; see --help

# bayes_1d/ — prior/likelihood/posterior plots (bayes_gaussian_1d*.png); see --help for prior/observation values
uv run python -m linear_kf.bayes_1d.bayes_gaussian_1d
```

`imu_1d/` is import-safe and writes into `--output-dir` (default `outputs/linear_kf/imu_1d/`, git-ignored):

```bash
uv run python -m linear_kf.imu_1d.demo
uv run python -m linear_kf.imu_1d.demo --scenario stop_and_go   # sine (default), stationary, stop_and_go
uv run python -m linear_kf.imu_1d.demo --outage 30 50 --accel-bias 0.05
uv run python -m linear_kf.imu_1d.make_doc_figures   # regenerates imu_1d/doc/*.png
uv run python -m linear_kf.imu_1d.animate            # movie of predict (variance grows) / update (shrinks) -> kf_imu1d_anim.mp4 + kf_imu1d_anim_error.png
```

```bash
# Wrong: these fail
python linear_kf/simple/kf_pv.py
cd linear_kf/simple && python kf_pv.py
```

### Where output goes

Every script writes its PNGs into the **current working directory**, not next to the script. Running
from the repo root is fine: `.gitignore` ignores `/*.png` there. To send the output somewhere else, run
from that directory and put the repo on `PYTHONPATH`:

```bash
mkdir -p /tmp/kf_out && cd /tmp/kf_out
PYTHONPATH=<repo root> uv run --project <repo root> python -m linear_kf.simple.kf_pv
```

`kf_pv_pykalman` and `kf_pv_em_pykalman` both write `kf_1d_state_var_pykalman.png`, so whichever runs
last overwrites the other.

### `kf_pva3d_pykalman` (offline estimation from a CSV)

This one is a CLI. It takes a parameter JSON file and an observation CSV file
(`time,x,y,z,s_xx,s_yy,s_zz,s_xy,s_xz,s_yz`), and writes `out_kf_3d_*_pykalman.png`:

```bash
uv run python -m linear_kf.pykalman_check.kf_pva3d_pykalman sample_kf_pva3d.json pos3d_obs.csv [-T]
```

`-T` shifts time so it starts at 0. The repo root has a sample parameter file
(`sample_kf_pva3d.json`) but no observation CSV. To generate one, call `run_test()`. It writes
`pos3d_obs.csv`, `param_kf_pva3d.json` and the `kf_3d_*_pykalman.png` plots into the current directory:

```bash
uv run python -c "from linear_kf.pykalman_check.kf_pva3d_pykalman import run_test; run_test()"
```

Git does not ignore the `.csv` and `.json` files that `run_test()` writes, so don't commit them by accident.

## Tests

Only `x_generator.py`, `simple/kf_pva3d.py`, `bayes_1d/` and `imu_1d/` are safe to import. The other scripts
run and save plots as soon as they are imported. The tests cover just those modules:

```bash
uv run pytest tests/linear_kf
```
