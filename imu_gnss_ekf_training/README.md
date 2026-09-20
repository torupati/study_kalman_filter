# 2D IMU/GNSS EKF training project

This directory contains a NumPy-only extended Kalman filter (EKF) training example for planar self-localization with IMU and GNSS.

## State, input, and measurement models

The EKF state is

\[
\mathbf{x} = [x,\ y,\ v_x,\ v_y,\ \psi,\ b_{ax},\ b_{ay},\ b_g]^T
\]

with body-frame IMU inputs

\[
\mathbf{u} = [a_x^m,\ a_y^m,\ \omega_z^m]^T
\]

and GNSS measurements

\[
\mathbf{z} = [x^{GNSS},\ y^{GNSS},\ v_x^{GNSS},\ v_y^{GNSS}]^T.
\]

The continuous-time training model is

- \(\dot{x} = v_x\)
- \(\dot{y} = v_y\)
- \(\dot{\mathbf{v}} = R(\psi) (\mathbf{a}^m - \mathbf{b}_a)\)
- \(\dot{\psi} = \omega_z^m - b_g\)
- \(\dot{b}_{ax}, \dot{b}_{ay}, \dot{b}_{g}\) are random walks

where

\[
R(\psi) =
\begin{bmatrix}
\cos\psi & -\sin\psi \\
\sin\psi & \cos\psi
\end{bmatrix}.
\]

The implementation in `ekf.py` discretizes this model, propagates the covariance with the EKF Jacobian, and normalizes yaw to `[-pi, pi)` after prediction and update.

## Directory structure

- `simulator.py` – truth trajectory, IMU, and GNSS simulation with configurable dropout and noise
- `ekf.py` – EKF predict/update logic, Jacobians, and bias random-walk process noise
- `plotting.py` – reusable plotting helpers for trajectories, errors, and bias estimates
- `demo.py` – runnable end-to-end simulator + EKF demo
- `experiments.py` – simple experiment runner for comparing GNSS dropout cases

## Run the demo

From the repository root:

```bash
python -m imu_gnss_ekf_training.demo
```

Optional arguments:

```bash
python -m imu_gnss_ekf_training.demo --total-time 90 --gnss-dropout 0.35 --output-dir outputs/demo_run
```

The demo prints RMSE metrics and saves `ekf_summary.png` with:

- truth vs estimated trajectory
- position errors
- velocity errors
- yaw error
- estimated accelerometer bias vs truth
- estimated gyro bias vs truth

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
