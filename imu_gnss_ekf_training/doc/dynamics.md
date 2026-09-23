# Process model and EKF linearization

This note derives the process (dynamic) model used by the planar IMU/GNSS EKF
and shows how it is linearized into the error-state Jacobians `F` and `G`
implemented in [`ekf.py`](../ekf.py) (`ImuGnssEkf.predict`). The measurement
update (`H`, `R`, Kalman gain) is documented separately.

## 1. State and input

The state vector is

$$
\mathbf{x} = [x,\ y,\ v_x,\ v_y,\ \psi,\ b_{ax},\ b_{ay},\ b_g]^T
$$

(position, velocity, yaw, and the accelerometer/gyro biases — see `IDX_*` in
`ekf.py`). The IMU provides body-frame control inputs

$$
\mathbf{u} = [a_x^m,\ a_y^m,\ \omega^m]^T .
$$

## 2. Continuous-time nonlinear dynamics

Modeling the accelerometer/gyro measurements as the true body-frame
acceleration/rate corrupted by a slowly-varying bias and white noise,

$$
a_x^m = a_x + b_{ax} + n_{ax}, \quad
a_y^m = a_y + b_{ay} + n_{ay}, \quad
\omega^m = \omega + b_g + n_g,
$$

the continuous-time process model is

$$
\dot{x} = v_x, \qquad \dot{y} = v_y
$$

$$
\dot{\mathbf{v}} = R(\psi)\, \mathbf{a}_c, \qquad
\mathbf{a}_c = \mathbf{a}^m - \mathbf{b}_a
= \begin{bmatrix} a_x^m - b_{ax} \\ a_y^m - b_{ay} \end{bmatrix}
$$

$$
\dot{\psi} = \omega^m - b_g
$$

$$
\dot{b}_{ax} = n_{bax}, \qquad \dot{b}_{ay} = n_{bay}, \qquad \dot{b}_g = n_{bg}
$$

where $\mathbf{a}_c$ is the *bias-corrected* body-frame acceleration used as
the control input, $n_{bax}, n_{bay}, n_{bg}$ are random-walk driving noises
for the biases, and

$$
R(\psi) =
\begin{bmatrix}
\cos\psi & -\sin\psi \\
\sin\psi & \cos\psi
\end{bmatrix}
$$

rotates the corrected body-frame acceleration into the navigation frame.
Biases are modeled as random walks because they drift slowly rather than
following any known deterministic law — the filter can only track them
through the process noise term, not the nominal (noise-free) propagation.

Again, using state vector $\mathbf{x}(t)$, the dynamics separates cleanly into a
deterministic (nominal) part $f(\mathbf{x}, \mathbf{u})$ and a stochastic
(noise-injection) part $L\,\mathbf{w}$, $\dot{\mathbf{x}} = f(\mathbf{x}, \mathbf{u}) + L\,\mathbf{w}$:

$$
\frac{d}{dt}\left[
\begin{matrix}
    x \\
    y\\
    v_x\\
    v_y \\
    \psi \\
    b_{ax} \\
    b_{ay} \\
    b_{g}
\end{matrix}
\right]
=
\underbrace{
\left[
\begin{matrix}
    v_x \\
    v_y\\
    \cos\psi (a_x^m - b_{ax}) - \sin \psi (a_y^m - b_{ay})\\
    \sin\psi (a_x^m - b_{ax}) + \cos \psi (a_y^m - b_{ay}) \\
    \omega^m - b_{g}\\
    0 \\
    0 \\
    0
\end{matrix}
\right]
}_{f(\mathbf{x}, \mathbf{u})\ (\text{nominal} = \S2\text{'s } a_c \text{ dynamics})}
+
\underbrace{
\left[
\begin{matrix}
    0 \\
    0\\
    -\cos\psi\, n_{ax} + \sin \psi\, n_{ay}\\
    -\sin\psi\, n_{ax} - \cos \psi\, n_{ay} \\
    -\, n_{g}\\
    n_{bax} \\
    n_{bay} \\
    n_{bg}
\end{matrix}
\right]
}_{L\,\mathbf{w}\ (\text{noise injection})}
$$

with $\mathbf{w} = [n_{ax},\ n_{ay},\ n_g,\ n_{bax},\ n_{bay},\ n_{bg}]^T$, the same
noise vector used in §6. The first term is exactly the nominal map evaluated at
$a_c = a^m - b_a$ (§2/§3, noise dropped since it's unknown at runtime); the
second is what remains after substituting $a^m = a + b_a + n_a$ (§2) and
cancelling the bias terms — accelerometer noise enters velocity rotated by
$R(\psi)$ (same as the signal), gyro noise enters yaw rate directly, and each
bias noise drives its own random walk directly.

Noise statistics (white, zero-mean, mutually independent):

$$
E[n_{ax}(t)] = E[n_{ay}(t)] = E[n_g(t)] = 0, \qquad
E[n_{ax}(t)\,n_{ax}(t')] = \sigma_{ax}^2\,\delta(t-t'), \quad
E[n_{ay}(t)\,n_{ay}(t')] = \sigma_{ay}^2\,\delta(t-t'), \quad
E[n_g(t)\,n_g(t')] = \sigma_g^2\,\delta(t-t')
$$

$$
E[n_{bax}(t)] = E[n_{bay}(t)] = E[n_{bg}(t)] = 0, \qquad
E[n_{bax}(t)\,n_{bax}(t')] = \sigma_{bax}^2\,\delta(t-t'), \quad
E[n_{bay}(t)\,n_{bay}(t')] = \sigma_{bay}^2\,\delta(t-t'), \quad
E[n_{bg}(t)\,n_{bg}(t')] = \sigma_{bg}^2\,\delta(t-t')
$$

where $\sigma_{ax}, \sigma_{ay}, \sigma_g$ are the IMU measurement-noise stds
(`accel_noise_std`, `gyro_noise_std`) and $\sigma_{bax}, \sigma_{bay}, \sigma_{bg}$
are the bias random-walk stds (`accel_bias_walk_std`, `gyro_bias_walk_std`),
matching §6's `Q` construction.


## 3. Discrete-time nominal propagation

Given a time step `dt`, `predict()` integrates the nominal (noise-free)
state with a second-order update for position (constant-acceleration
assumption over the step) and a first-order update for velocity:

$$
\mathbf{a}_{nav} = R(\psi_k)\, \mathbf{a}_c
$$

$$
x_{k+1} = x_k + v_{x,k}\, dt + \tfrac{1}{2} a_{nav,x}\, dt^2, \qquad
y_{k+1} = y_k + v_{y,k}\, dt + \tfrac{1}{2} a_{nav,y}\, dt^2
$$

$$
v_{x,k+1} = v_{x,k} + a_{nav,x}\, dt, \qquad
v_{y,k+1} = v_{y,k} + a_{nav,y}\, dt
$$

$$
\psi_{k+1} = \operatorname{wrap}\!\big(\psi_k + (\omega^m - b_{g,k})\, dt\big)
$$

$$
b_{ax,k+1} = b_{ax,k}, \qquad b_{ay,k+1} = b_{ay,k}, \qquad b_{g,k+1} = b_{g,k}
$$

(the biases are held constant in the nominal propagation; their random-walk
uncertainty only enters through the process noise covariance `Q` below).
`wrap(·)` normalizes the angle to `[-pi, pi)` via `atan2(sin, cos)`, matching
`ekf.py`'s use of `np.arctan2` after both `predict` and `update`.

This nominal map is exactly the function $\mathbf{x}_{k+1} = f(\mathbf{x}_k, \mathbf{u}_k)$
that `predict()` evaluates, and it is also what `simulator.py` uses to
propagate ground truth (with $b_a, b_g$ replaced by the true, separately
simulated bias random walks) — see the note in the top-level
[`CLAUDE.md`](../../CLAUDE.md) that changes to the process model must be
mirrored in both files.

## 4. Error-state model

Because $f$ is nonlinear in $\psi$ (through $R(\psi)$), the EKF does not
propagate a probability distribution over $\mathbf{x}$ exactly. Instead it
tracks a point estimate $\hat{\mathbf{x}}_k$ and linearizes the dynamics of
the *error*

$$
\delta\mathbf{x}_k = \mathbf{x}_k - \hat{\mathbf{x}}_k
$$

around the current estimate. Writing the true state as
$\hat{\mathbf{x}}_k + \delta\mathbf{x}_k$ and Taylor-expanding
$f(\mathbf{x}_k, \mathbf{u}_k)$ to first order in $\delta\mathbf{x}_k$ and in
the process noise $\mathbf{w}_k$,

$$
\delta\mathbf{x}_{k+1} \approx F_k\, \delta\mathbf{x}_k + G_k\, \mathbf{w}_k,
\qquad
F_k = \left.\frac{\partial f}{\partial \mathbf{x}}\right|_{\hat{\mathbf{x}}_k, \mathbf{u}_k},
\qquad
G_k = \left.\frac{\partial f}{\partial \mathbf{w}}\right|_{\hat{\mathbf{x}}_k, \mathbf{u}_k}.
$$

The covariance of $\delta\mathbf{x}_{k+1}$ is then propagated linearly,

$$
P_{k+1} = F_k P_k F_k^T + Q_k, \qquad Q_k = G_k\, \mathrm{cov}(\mathbf{w}_k)\, G_k^T,
$$

which is exactly `predicted_covariance = F @ covariance @ F.T + Q` in
`predict()` (followed by the `0.5 * (P + P.T)` symmetrization to counter
numerical drift). The state itself is *not* linear — $\hat{\mathbf{x}}_{k+1}$
is still produced by evaluating the nonlinear $f$ directly — only its
uncertainty is propagated through the linearized $F_k$, which is the
defining trait of an EKF as opposed to a linear KF.

## 5. Computing the state Jacobian F

Only the position/velocity/yaw rows depend on the state nonlinearly (through
$\psi$ and the biases entering $\mathbf{a}_c$); the rest of $f$ is already
linear or constant, so those rows of $F$ are just the identity plus the `dt`
position/velocity coupling. Let

$$
a_{x}^c = a_x^m - b_{ax}, \qquad a_{y}^c = a_y^m - b_{ay},
$$

so that $a_{nav,x} = c\, a_x^c - s\, a_y^c$ and $a_{nav,y} = s\, a_x^c + c\, a_y^c$
with $c = \cos\psi_k$, $s = \sin\psi_k$. Differentiating with respect to
$\psi$ (holding $a_x^c, a_y^c$ fixed, since they don't depend on $\psi$):

$$
\frac{\partial a_{nav,x}}{\partial \psi} = -s\, a_x^c - c\, a_y^c, \qquad
\frac{\partial a_{nav,y}}{\partial \psi} = c\, a_x^c - s\, a_y^c,
$$

which are exactly `dax_dyaw` and `day_dyaw` in the code. Differentiating with
respect to the biases (note $\partial a_x^c/\partial b_{ax} = -1$,
$\partial a_y^c/\partial b_{ay} = -1$):

$$
\frac{\partial a_{nav,x}}{\partial b_{ax}} = -c, \quad
\frac{\partial a_{nav,x}}{\partial b_{ay}} = s, \quad
\frac{\partial a_{nav,y}}{\partial b_{ax}} = -s, \quad
\frac{\partial a_{nav,y}}{\partial b_{ay}} = -c.
$$

Propagating these through the position ($\tfrac12 dt^2 \cdot$) and velocity
($dt \cdot$) update equations gives every off-diagonal entry of `F` built in
`predict()`:

| entry | value | meaning |
|---|---|---|
| `F[x, vx]`, `F[y, vy]` | `dt` | position ← velocity |
| `F[x, ψ]`, `F[y, ψ]` | `0.5*dt²·∂a_nav/∂ψ` | position ← yaw (via rotated accel) |
| `F[vx, ψ]`, `F[vy, ψ]` | `dt·∂a_nav/∂ψ` | velocity ← yaw |
| `F[x, bax]`, `F[x, bay]` | `∓0.5*dt²·c`, `±0.5*dt²·s` | position ← accel bias |
| `F[y, bax]`, `F[y, bay]` | `-0.5*dt²·s`, `-0.5*dt²·c` | position ← accel bias |
| `F[vx, bax]`, `F[vx, bay]` | `-dt·c`, `dt·s` | velocity ← accel bias |
| `F[vy, bax]`, `F[vy, bay]` | `-dt·s`, `-dt·c` | velocity ← accel bias |
| `F[ψ, bg]` | `-dt` | yaw ← gyro bias |

All remaining entries are the identity ($\partial x_{k+1}/\partial x_k = 1$
for every state, plus the `dt` position/velocity coupling above) — biases are
modeled as constant in the nominal map, so their own diagonal entries are 1
and they otherwise only appear in the columns above.

## 6. Process noise Jacobian G and Q

The noise vector driving the discrete step is

$$
\mathbf{w} = [n_{ax},\ n_{ay},\ n_g,\ n_{bax},\ n_{bay},\ n_{bg}]^T,
$$

i.e. the two accelerometer noise components, the gyro noise, and the three
bias random-walk increments. Since $a_{nav} = R(\psi)(\mathbf{a}^m - \mathbf{b}_a)$
and the noise enters additively through $\mathbf{a}^m$, $\partial a_{nav}/\partial(n_{ax}, n_{ay}) = R(\psi)$
directly (the *full* rotation, unlike the bias columns of `F` above, because
noise perturbs the measurement itself rather than being subtracted off):

$$
G[[x,y],\, 0{:}2] = \tfrac12 dt^2\, R(\psi), \qquad
G[[v_x,v_y],\, 0{:}2] = dt\, R(\psi), \qquad
G[\psi,\, 2] = dt,
$$

matching the `predict()` code. The bias rows map each random-walk driving
noise directly onto the corresponding bias, scaled by $\sqrt{dt}$ so that the
discrete increment has the correct variance for a continuous-time random walk
with density (per-`sqrt(second)` std) `*_walk_std`:

$$
G[b_{ax},\, 3] = G[b_{ay},\, 4] = G[b_g,\, 5] = \sqrt{dt}.
$$

`Q = G @ diag(σ_ax², σ_ay², σ_g², σ_bax_walk², σ_bay_walk², σ_bg_walk²) @ G.T`
then combines per-sample IMU measurement noise (`accel_noise_std`,
`gyro_noise_std`) with the bias random-walk increment variance
(`accel_bias_walk_std² · dt`, `gyro_bias_walk_std² · dt`, folded into `Q`
through the `sqrt(dt)` factors in `G`) into the full `STATE_SIZE × STATE_SIZE`
process noise covariance added to `P` each step.

## Summary

`predict()` implements exactly this two-part linearization: the nominal state
$\hat{\mathbf{x}}_{k+1}$ is produced by the nonlinear map $f$ in §3, while the
covariance is propagated using the Jacobians $F$ (§5) and $G$ (§6) evaluated
at the current estimate — the standard continuous-dynamics/discrete-EKF
recipe of linearizing only the *error* state, not the nominal trajectory.
