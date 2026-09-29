# 2D IMU/GNSS EKF training project

This directory contains a NumPy-only extended Kalman filter (EKF) training example for planar self-localization with IMU and GNSS.

## State, input, and measurement models

The EKF state is

$$
\mathbf{x} = [x,\ y,\ v_x,\ v_y,\ \psi,\ b_{ax},\ b_{ay},\ b_g]^T
$$

with body-frame IMU inputs

$$
\mathbf{u} = [a_x^m,\ a_y^m,\ \omega_z^m]^T
$$

and GNSS measurements

$$
\mathbf{z} = [x^{GNSS},\ y^{GNSS},\ v_x^{GNSS},\ v_y^{GNSS}]^T.
$$

The continuous-time process model is

- $\dot{x} = v_x$
- $\dot{y} = v_y$
- $\dot{\mathbf{v}} = R(\psi) (\mathbf{a}^m - \mathbf{b}_a)$
- $\dot{\psi} = \omega_z^m - b_g$
- $\dot{b}_{ax}, \dot{b}_{ay}, \dot{b}_{g}$ are random walks

where

$$
R(\psi) =
\begin{bmatrix}
\cos\psi & -\sin\psi \\
\sin\psi & \cos\psi
\end{bmatrix}.
$$

The implementation in `ekf.py` discretizes this model, propagates the covariance with the EKF Jacobian, and normalizes yaw to `[-pi, pi)` after prediction and update.

## Directory structure

- `simulator.py` – truth trajectory, IMU, and GNSS simulation with configurable dropout and noise
- `ekf.py` – EKF predict/update logic, Jacobians, and bias random-walk process noise
- `plotting.py` – reusable plotting helpers for trajectories, errors, and bias estimates
- `demo.py` – runnable end-to-end simulator + EKF demo
- `experiments.py` – simple experiment runner for comparing GNSS dropout cases
- `navlog.py` – `NavLog`: one EKF run (posterior/prior states and covariances, GNSS, innovations, optional truth) saved as `.npz`
- `ellipse.py` – covariance → confidence-ellipse geometry (pure NumPy)
- `animation.py` / `animate.py` – render a `NavLog` as a movie: the 2D trajectory with covariance ellipses and GNSS fixes (`--view trajectory`), or bias estimates vs. truth (`--view bias`)
- `doc/` – notes: [coordinate frames](doc/coordinate_frames.md), [process model](doc/dynamics.md), [process noise Q](doc/process_noise.md), [math background](doc/preliminary_math.md), [bias estimation convergence by scenario](doc/bias_convergence.md), [visual odometry (delta pose) simulation](doc/visual_odometry.md), [IMU + VO navigation without GNSS](doc/vo_only_navigation.md)

## Run the demo

From the repository root:

```bash
python -m imu_gnss_ekf_training.demo
```

Optional arguments:

```bash
python -m imu_gnss_ekf_training.demo --total-time 90 --gnss-dropout 0.35 --output-dir outputs/demo_run
python -m imu_gnss_ekf_training.demo --scenario circle --dt 0.01 --use-vo   # also fuse 10 fps visual odometry
python -m imu_gnss_ekf_training.demo --scenario demo2 --dt 0.01 --use-vo --no-gnss-update --initial-pose 30 -20 40   # IMU+VO only; see doc/visual_odometry.md §8
```

The demo prints RMSE metrics and saves `ekf_summary.png` with:

- truth vs estimated trajectory
- position errors
- velocity errors
- yaw error
- estimated accelerometer bias vs truth
- estimated gyro bias vs truth

## Make a navigation movie

Save a navigation log from a run, then render it (post-processing; the filter is not re-run):

```bash
python -m imu_gnss_ekf_training.demo --save-log outputs/run1/nav_log.npz
python -m imu_gnss_ekf_training.run_ekf_from_csv --save-log outputs/run1/nav_log.npz   # recorded CSV data
python -m imu_gnss_ekf_training.animate outputs/run1/nav_log.npz --output outputs/run1/nav.mp4 --speed 2
python -m imu_gnss_ekf_training.animate outputs/run1/nav_log.npz --output outputs/run1/nav.gif --follow 15 --trail-seconds 10
python -m imu_gnss_ekf_training.animate outputs/run1/nav_log.npz --view bias --output outputs/run1/bias.mp4
```

Each frame shows the truth/EKF trails, GNSS fixes, the posterior position ellipse (filled), the
prior ellipse at the last GNSS update (dashed), the GNSS measurement-noise ellipse `R` (dotted), the
heading with its yaw confidence fan, and a lower panel of ellipse semi-major axis vs. position error.
`--view bias` instead draws gyro z, accel x, and accel y bias in three panels on a fixed time axis:
true bias, estimate, and the estimate's confidence band.
Ellipses are drawn at a probability level (`--confidence`, default 95%): in 2D the radius is
`k = sqrt(-2 ln(1 - p))` sigma, so a "1-sigma" ellipse contains only ~39%. `.mp4` needs `ffmpeg`;
`.gif` works without it.

## Simulation scenarios

`demo.py --scenario NAME` (and `record_sensors.py NAME`) selects the truth trajectory:

| scenario | motion | suggested `--total-time` |
|---|---|---|
| `demo1` | smooth, continuously curving (default) | 60 |
| `demo2` | accelerate, 30 m straight, half-circle turn, straight back, stop | 60 |
| `stationary` | stays at the origin | 30 |
| `forward_back` | hold 3 s, 5 m forward along +x, hold 3 s, reverse 5 m back to the origin (no turning); distance set by `--forward-back-distance` | 30 |
| `circle` | from rest, speed ramps to 2 m/s on a counter-clockwise 10 m radius circle centered at (0, 10) | 40 |

`./imu_gnss_ekf_training/run_scenario_movies.sh` runs `stationary`, `forward_back`, and `circle` and writes
`outputs/scenarios/<scenario>/nav.mp4` and `bias.mp4` for each (set `FPS=...` to trade smoothness for render time).
The true initial bias is set with `--initial-accel-bias BX BY` [m/s²] and `--initial-gyro-bias-dps` [deg/s].
See [doc/bias_convergence.md](doc/bias_convergence.md) for how each scenario lets the biases converge.

## Run the dropout experiment

```bash
python -m imu_gnss_ekf_training.experiments
```

This produces `dropout_experiments.png` to compare dense and intermittent GNSS updates.

## Training flow extension ideas

A natural stepwise teaching flow is:

1. position/velocity-only linear KF
2. add accelerometer bias states
3. add yaw and gyro bias
4. replace the linear transition with the nonlinear IMU rotation model
5. study the EKF Jacobian and GNSS dropout behavior
