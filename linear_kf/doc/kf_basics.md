# Linear Kalman filter basics: the 1D position/velocity model

This document derives the linear Kalman filter (KF) and the Rauch–Tung–Striebel
(RTS) smoother, and maps every equation onto the from-scratch implementation in
[`simple/kf_pv.py`](../simple/kf_pv.py) and its `pykalman` counterpart
[`pykalman_check/kf_pv_pykalman.py`](../pykalman_check/kf_pv_pykalman.py).
It is the entry point for the `linear_kf/` documents; the extended filter in
[`imu_gnss_ekf_training/`](../../imu_gnss_ekf_training/README.md) uses the same
predict/update structure with Jacobians in place of $F$ and $H$.

Background on covariance propagation, white noise and conditional Gaussians is
in [`imu_gnss_ekf_training/doc/preliminary_math.md`](../../imu_gnss_ekf_training/doc/preliminary_math.md).
The process noise matrix $Q$ used below is derived and checked by Monte Carlo
in [`process_noise_sim/doc/process_noise.md`](../process_noise_sim/doc/process_noise.md)
(Model 1).

## 1. The model

A linear Gaussian state-space model has two equations:

$$
\mathbf{x}_{k+1} = F_k\,\mathbf{x}_k + \mathbf{w}_k, \qquad \mathbf{w}_k \sim \mathcal{N}(\mathbf{0},\, Q_k),
$$

$$
\mathbf{y}_k = H\,\mathbf{x}_k + \mathbf{v}_k, \qquad \mathbf{v}_k \sim \mathcal{N}(\mathbf{0},\, R),
$$

with the initial state $\mathbf{x}_0 \sim \mathcal{N}(\hat{\mathbf{x}}_0,\, P_0)$ and
$\mathbf{w}_k$, $\mathbf{v}_k$, $\mathbf{x}_0$ mutually independent.

- $F_k$ (state transition) says how the state moves on its own over one step.
- $Q_k$ (process noise covariance) says how much the true motion departs from
  that prediction in one step.
- $H$ (observation matrix) picks out what the sensor measures.
- $R$ (observation noise covariance) says how noisy the sensor is.

Because everything is linear and Gaussian, the conditional distribution of the
state given the measurements is Gaussian too, so it is fully described by a
mean $\hat{\mathbf{x}}$ and a covariance $P$. The KF is the recursion that
updates these two quantities.

### 1.1 The 1D position/velocity model in `kf_pv.py`

The state is position and velocity, $\mathbf{x} = [p,\ v]^T$, and only position
is measured. Acceleration is modelled as continuous-time white noise with power
spectral density (PSD) $q = \sigma_a^2$ (the *random acceleration* or
*constant-velocity* model):

$$
\dot p = v, \qquad \dot v = w(t), \qquad E[w(t)\,w(s)] = \sigma_a^2\,\delta(t-s).
$$

Discretizing over a step $\Delta t$ gives

$$
F = \begin{bmatrix} 1 & \Delta t \\ 0 & 1 \end{bmatrix}, \qquad
Q = \sigma_a^2 \begin{bmatrix} \Delta t^3/3 & \Delta t^2/2 \\ \Delta t^2/2 & \Delta t \end{bmatrix}, \qquad
H = \begin{bmatrix} 1 & 0 \end{bmatrix}, \qquad
R = \sigma_p^2 .
$$

$F$ is exact for this model (constant velocity between samples), and $Q$ is the
exact covariance of the integrated white noise, not an approximation; see
`process_noise.md` Model 1 for the integral. Note the units: $\sigma_a$ is a
noise *density* in $\mathrm{m/s^2/\sqrt{Hz}}$ (equivalently
$\mathrm{m\,s^{-3/2}}$), so $\mathrm{Var}[v]$ grows like $\sigma_a^2\,\Delta t$.

The values come from `kf_pv_1d_type1_condition()` in
[`x_generator.py`](../x_generator.py):

| Symbol | Code | Value | Meaning |
|---|---|---|---|
| $1/\Delta t$ | `Fs` | 2 Hz | sample rate |
| — | `t_end` | 60 s | simulation length |
| $\sigma_a$ | `sig0` (`sig_acc`) | 1.0 | acceleration noise density |
| $\sigma_p$ | `sig1` (`sig_pos`) | 0.5 m | position measurement noise std |
| $\hat{\mathbf{x}}_0$ | `x_init` | $[0,\ 0]^T$ | initial mean |
| $P_0$ | `x_var_init` | $\mathrm{diag}(4, 4)$ | initial covariance |

| Matrix | `kf_pv.py` | `kf_pv_pykalman.py` (`KalmanFilter(...)` argument) |
|---|---|---|
| $F$ | `F` | `transition_matrices` |
| $Q$ | `Q` | `transition_covariance` |
| $H$ | `H` | `observation_matrices` |
| $R$ | `R` | `observation_covariance` |
| $\hat{\mathbf{x}}_0,\ P_0$ | `x_init`, `x_var_init` | `initial_state_mean`, `initial_state_covariance` |

The truth trajectory (`generte_true_pos_vel_1d_type1`) is a sinusoid,
$p(t) = 2.3\,(\cos\omega t - 1)$ with a period of `t_end / 2` = 30 s, not a sample of the random
acceleration model. So the filter runs with a deliberately *mismatched* model:
$\sigma_a$ has to be large enough to cover the true acceleration
($2.3\,\omega^2 \approx 0.10\ \mathrm{m/s^2}$ peak), which is the usual
situation in practice.

## 2. Notation

$\hat{\mathbf{x}}_{k|j}$ and $P_{k|j}$ are the mean and covariance of
$\mathbf{x}_k$ given measurements $\mathbf{y}_0, \dots, \mathbf{y}_j$.

- $\hat{\mathbf{x}}_{k|k-1}$, $P_{k|k-1}$: **prior** (predicted), before using $\mathbf{y}_k$.
- $\hat{\mathbf{x}}_{k|k}$, $P_{k|k}$: **posterior** (filtered), after using $\mathbf{y}_k$.
- $\hat{\mathbf{x}}_{k|N}$, $P_{k|N}$: **smoothed**, using all $N+1$ measurements.

## 3. Prediction (time update)

Take the mean and covariance of $\mathbf{x}_{k+1} = F\mathbf{x}_k + \mathbf{w}_k$
given $\mathbf{y}_{0:k}$. Since $\mathbf{w}_k$ is zero-mean and independent of
$\mathbf{x}_k$ (preliminary_math §2):

$$
\hat{\mathbf{x}}_{k+1|k} = F\,\hat{\mathbf{x}}_{k|k}, \qquad
P_{k+1|k} = F\,P_{k|k}\,F^T + Q .
$$

The mean moves with the model, and the covariance is transported by $F$
and then grows by $Q$. With no measurements, the prediction alone makes $P$
grow without bound: for this model the position variance grows like
$\sigma_a^2 t^3 / 3$.

In `kf_pv.py`:

```python
x = np.dot(F, x)
P = np.dot(np.dot(F, P), F.T) + Q
```

## 4. Update (measurement update)

Before seeing $\mathbf{y}_k$, the state and the predicted measurement are jointly Gaussian:

$$
\begin{bmatrix} \mathbf{x}_k \\ \mathbf{y}_k \end{bmatrix} \sim
\mathcal{N}\!\left(
\begin{bmatrix} \hat{\mathbf{x}}_{k|k-1} \\ H\hat{\mathbf{x}}_{k|k-1} \end{bmatrix},\
\begin{bmatrix} P_{k|k-1} & P_{k|k-1}H^T \\ H P_{k|k-1} & S_k \end{bmatrix}
\right),
\qquad S_k = H P_{k|k-1} H^T + R .
$$

Conditioning a joint Gaussian on one block (preliminary_math §6) gives the posterior directly:

$$
\boxed{
\begin{aligned}
\mathbf{e}_k &= \mathbf{y}_k - H\,\hat{\mathbf{x}}_{k|k-1} && \text{innovation} \\
S_k &= H P_{k|k-1} H^T + R && \text{innovation covariance} \\
K_k &= P_{k|k-1} H^T S_k^{-1} && \text{Kalman gain} \\
\hat{\mathbf{x}}_{k|k} &= \hat{\mathbf{x}}_{k|k-1} + K_k\,\mathbf{e}_k \\
P_{k|k} &= (I - K_k H)\,P_{k|k-1}
\end{aligned}
}
$$

In `kf_pv.py`:

```python
S = np.dot(np.dot(H, P), H.T) + R
K = np.dot(np.dot(P, H.T), np.linalg.inv(S))
x = x + np.dot(K, y - np.dot(H, x))
P = np.dot((np.eye(2) - np.dot(K, H)), P)
```

### 4.1 What the gain does

For this model $S_k = P_{pp} + \sigma_p^2$ is a scalar (with $P_{pp}$, $P_{pv}$
the elements of $P_{k|k-1}$) and

$$
K_k = \frac{1}{P_{pp} + \sigma_p^2}\begin{bmatrix} P_{pp} \\ P_{pv} \end{bmatrix} .
$$

- The position gain $P_{pp}/(P_{pp}+\sigma_p^2)$ is a weight between 0 and 1: it
  is close to 1 when the prediction is much less certain than the sensor, and
  close to 0 when the sensor is much noisier than the prediction.
- Velocity is never measured, but it still gets corrected, through the
  cross-covariance $P_{pv}$. The prediction step creates that correlation
  ($F$ mixes $v$ into $p$, and $Q$ has off-diagonal $\sigma_a^2\Delta t^2/2$),
  and the update uses it. This is how a KF estimates states it cannot see.

### 4.2 Steady state

$F$, $H$, $Q$ and $R$ are constant here, so $P_{k|k-1}$ converges to a fixed
matrix (the solution of the discrete algebraic Riccati equation) and $K_k$ to a
constant gain. The `kf_1d_state_var.png` plot shows this: after the first few
updates shrink $P_0 = \mathrm{diag}(4,4)$, the filtered variances are flat.
The steady state depends on $\sigma_a$ and $\sigma_p$ only through their ratio
(scaled by $\Delta t$); doubling both doubles every standard deviation but leaves $K$
unchanged.

### 4.3 Numerically safer forms

The short form $P = (I - KH)P$ is exact algebra but loses symmetry and positive
definiteness under rounding when $K$ is not exactly optimal. The IMU/GNSS EKF in
`imu_gnss_ekf_training/ekf.py` uses the Joseph form,

$$
P_{k|k} = (I - K_k H)\,P_{k|k-1}\,(I - K_k H)^T + K_k R K_k^T ,
$$

which stays symmetric positive semi-definite for any $K$, and computes
$K$ with `np.linalg.solve` instead of an explicit inverse. For this 2-state
model with a scalar $S$ the difference is invisible, which is why `kf_pv.py`
keeps the textbook form.

## 5. Order of steps in the code

`kf_pv.py` treats $(\hat{\mathbf{x}}_0, P_0)$ as the **prior at the first
measurement time** $t_0$, so each loop iteration does *update, then predict*:

```text
x, P = x0, P0                      # prior at t_0
for k in 0..N:
    x, P = update(x, P, y_k)       # posterior at t_k, stored in x_est, P_est
    if k == N: break
    x, P = predict(x, P)           # prior at t_{k+1}
```

`pykalman`'s `KalmanFilter.filter` uses the same convention
(`initial_state_mean` is the prior for the first observation), which is why the
two scripts give the same filtered estimates for the same noise draw.

## 6. RTS smoother

The filter estimate at time $k$ uses only measurements up to $k$. For offline
analysis, all measurements are available, and the RTS smoother improves every
estimate with the later data. It runs **backwards** over the stored filter output:

$$
\begin{aligned}
\hat{\mathbf{x}}_{k+1|k} &= F\,\hat{\mathbf{x}}_{k|k}, \qquad P_{k+1|k} = F P_{k|k} F^T + Q \\
J_k &= P_{k|k}\,F^T\,P_{k+1|k}^{-1} && \text{smoother gain} \\
\hat{\mathbf{x}}_{k|N} &= \hat{\mathbf{x}}_{k|k} + J_k\,(\hat{\mathbf{x}}_{k+1|N} - \hat{\mathbf{x}}_{k+1|k}) \\
P_{k|N} &= P_{k|k} + J_k\,(P_{k+1|N} - P_{k+1|k})\,J_k^T
\end{aligned}
$$

starting from $\hat{\mathbf{x}}_{N|N}$, $P_{N|N}$ at the last step.

Intuition: $\hat{\mathbf{x}}_{k+1|N} - \hat{\mathbf{x}}_{k+1|k}$ is how much
the future data moved the estimate at $k+1$ away from what the filter predicted
at $k$. $J_k$ maps that correction back one step. Since
$P_{k+1|N} \preceq P_{k+1|k}$, the covariance correction is negative
semi-definite and $P_{k|N} \preceq P_{k|k}$: smoothing never makes the
estimate less certain. In `kf_1d_state_var.png` the smoothed variance is lowest
in the middle of the run, where data on both sides is available, and equals the
filtered variance at the last sample, where there is no future data.

In `kf_pv.py` the backward loop recomputes `_x_pred` and `_P_pred` from the
stored posteriors using the last `F` and `Q` from the forward loop. That
is correct here because $\Delta t$ is constant. With irregular sampling, the
forward pass should store each step's $F_k$, $Q_k$ (or the priors) for the smoother to reuse.
`pykalman`'s equivalent is `KalmanFilter.smooth`.

## 7. Output figures

Running `uv run python -m linear_kf.simple.kf_pv` from the repository root writes:

| File | Contents |
|---|---|
| `posvel_kf_pv_1d.png` | truth position/velocity and the noisy position measurements |
| `kf_1d_state_filter.png` | filtered position and velocity vs. truth |
| `kf_1d_state_smooth.png` | filtered and smoothed estimates vs. truth |
| `kf_1d_state_var.png` | diagonal of $P_{k\|k}$ and $P_{k\|N}$ over time |

`pykalman_check.kf_pv_pykalman` writes the `*_pykalman.png` versions of the
last three for comparison.

## 8. Known discrepancy in the measurement simulation

Both `simple/kf_pv.py` and `pykalman_check/kf_pv_pykalman.py` (and
`kf_pv_em_pykalman.py`) simulate the measurements as

```python
pos_obs = x_true[:, 0] + np.random.normal(0.0, sig1 * sig1, len(x_true))
```

`np.random.normal`'s second argument is the **standard deviation**, so the
simulated noise has std $\sigma_p^2 = 0.25$ m, while the filter assumes
$R = \sigma_p^2$, i.e. std $\sigma_p = 0.5$ m. The filter therefore assumes
twice the actual noise std (4× the variance). It still works, but it trusts the
measurements less than it should, so the estimates are smoother and lag more
than the optimal filter. Its reported $P$ is also pessimistic (larger than the actual error).
If EM were asked to estimate $R$ (`em_vars=['observation_covariance']` in
[`pykalman_check/kf_pv_em_pykalman.py`](../pykalman_check/kf_pv_em_pykalman.py),
which currently estimates only `initial_state_mean`), it should converge near
$0.25^2 = 0.0625$, not $0.25$, for this reason.
