# Process noise covariance $Q_k$

This is a supplement to [`dynamics.md`](dynamics.md) §4 and §6. It derives the
discrete process noise covariance $Q_k$ that `ImuGnssEkf.predict` adds to the
covariance each step,

$$
P_{k+1|k} = F_k P_{k|k} F_k^T + Q_k, \qquad Q_k = G_k\, \Sigma_w\, G_k^T ,
$$

and explains where each factor comes from. It covers what $Q_k$ means, the
exact formula, the simpler approximation the code uses, and what that
approximation leaves out. The background is in
[`preliminary_math.md`](preliminary_math.md) (§2 linear transformation of
covariance, §3 white noise). The integrated-white-noise formulas used below are
worked out and checked by Monte Carlo in
[`my_test_kf/doc/process_nosie.md`](../../my_test_kf/doc/process_nosie.md).

Notation follows [`dynamics.md`](dynamics.md). The error state is
$\delta\mathbf{x} = \mathbf{x} - \hat{\mathbf{x}}$, the noise vector is
$\mathbf{w} = [n_{ax},\ n_{ay},\ n_g,\ n_{bax},\ n_{bay},\ n_{bg}]^T$, and the
frames are defined in [`coordinate_frames.md`](coordinate_frames.md).

## 1. What $Q_k$ is

Over one step $t_k \to t_{k+1} = t_k + \Delta t$, the linearized error obeys

$$
\delta\mathbf{x}_{k+1} = F_k\, \delta\mathbf{x}_k + \mathbf{w}^d_k ,
$$

where $\mathbf{w}^d_k$ is the error that the noise adds during that step. It is
zero-mean and independent of $\delta\mathbf{x}_k$, so taking the covariance of
both sides (preliminary_math §2) gives

$$
P_{k+1|k} = F_k P_{k|k} F_k^T + Q_k, \qquad Q_k \equiv \operatorname{Cov}(\mathbf{w}^d_k) .
$$

So $Q_k$ is **the covariance of the error that the noise alone adds in one
step**. The only question is how to compute $\mathbf{w}^d_k$ from the noise
model.

## 2. The exact answer (Van Loan integral)

Linearizing the continuous-time model of dynamics §2 around the estimate gives

$$
\delta\dot{\mathbf{x}} = A\, \delta\mathbf{x} + L\, \mathbf{w}(t),
\qquad E[\mathbf{w}(t)\mathbf{w}(t')^T] = Q_c\, \delta(t - t') ,
$$

with $A = \partial f / \partial \mathbf{x}$ and $L$ the noise-injection matrix
(dynamics §2). $Q_c$ is the diagonal matrix of power spectral densities
(PSDs). Solving this linear ODE over the step,

$$
\delta\mathbf{x}_{k+1} = \underbrace{e^{A\Delta t}}_{\Phi(\Delta t)\,\approx\,F_k}\delta\mathbf{x}_k
+ \underbrace{\int_0^{\Delta t} \Phi(\Delta t - \tau)\, L\, \mathbf{w}(t_k + \tau)\, d\tau}_{\mathbf{w}^d_k} .
$$

Taking the covariance of $\mathbf{w}^d_k$ turns the double integral into a
single one, because $E[\mathbf{w}(\tau)\mathbf{w}(\tau')^T] = Q_c\,\delta(\tau-\tau')$:

$$
\boxed{\,Q_k^{\text{exact}} = \int_0^{\Delta t} \Phi(\tau)\, L\, Q_c\, L^T\, \Phi(\tau)^T\, d\tau\,}
$$

Van Loan's method evaluates this with one matrix exponential. Build the block
matrix $M = \begin{bmatrix} -A & L Q_c L^T \\ 0 & A^T \end{bmatrix}$ and compute
$e^{M\Delta t} = \begin{bmatrix} \cdot & E_{12} \\ 0 & E_{22} \end{bmatrix}$.
Then $\Phi = E_{22}^T$ and $Q_k^{\text{exact}} = \Phi\, E_{12}$.

This result is exact for the linearized model, but it produces a dense $8\times8$
matrix that depends on $\hat\psi$, $\mathbf{a}_c$ and $\Delta t$. The code
uses something simpler.

## 3. What the code does: $Q_k = G_k \Sigma_w G_k^T$

The code treats each noise source as a single random kick per step and pushes
it through the **discrete** update equations of dynamics §3. Its first-order
effect on the state is $G_k\,\mathbf{w}_k$ with $G_k = \partial f_d / \partial \mathbf{w}$
(dynamics §6). Then preliminary_math §2 gives $Q_k = G_k\, \operatorname{Cov}(\mathbf{w}_k)\, G_k^T$.

The code has two kinds of noise, and they need different treatment.

### 3.1 Bias random walks: $\sqrt{\Delta t}$ in $G$

A bias follows $\dot b = n_b(t)$, where $n_b$ is white noise with PSD $\sigma_{bw}^2$
(`accel_bias_walk_std`, `gyro_bias_walk_std`). Over one step,

$$
b_{k+1} - b_k = \int_0^{\Delta t} n_b(t_k+\tau)\, d\tau,
\qquad
\operatorname{Var}(b_{k+1} - b_k) = \int_0^{\Delta t}\!\!\int_0^{\Delta t} \sigma_{bw}^2\, \delta(\tau-\tau')\, d\tau\, d\tau' = \sigma_{bw}^2\, \Delta t .
$$

The increment's standard deviation therefore grows as $\sqrt{\Delta t}$, not
$\Delta t$. The code writes the increment as $\sqrt{\Delta t}\cdot n$ with
$\operatorname{Var}(n) = \sigma_{bw}^2$:

```python
G[IDX_BAX, 3] = np.sqrt(dt)   # also IDX_BAY, IDX_BG
```

This entry is **exact**. The units of $\sigma_{bw}$ follow from it:
m/s²/√s for the accelerometer and rad/s/√s for the gyro. `simulator.py`
generates the true biases the same way
(`bias += walk_std * sqrt(dt) * randn()`).

### 3.2 IMU measurement noise: per-sample noise held over the step

The accelerometer and gyro noise are handled differently. The code treats
$n_{ax}, n_{ay}, n_g$ as **one noise value per IMU sample**, with standard
deviation $\sigma_a$ (`accel_noise_std`, m/s²) or $\sigma_g$ (`gyro_noise_std`,
rad/s), **held constant for the whole step** (zero-order hold). This matches
`simulator.py`, which adds `std * randn()` to each IMU sample.

With a noise value $\mathbf{n}_a$ held over $\Delta t$, the discrete update
equations of dynamics §3 give its contribution directly:

$$
\delta\mathbf{p} \mathrel{+}= \tfrac12 \Delta t^2\, R(\psi)\, \mathbf{n}_a, \qquad
\delta\mathbf{v} \mathrel{+}= \Delta t\, R(\psi)\, \mathbf{n}_a, \qquad
\delta\psi \mathrel{+}= \Delta t\, n_g ,
$$

```python
G[[IDX_X, IDX_Y], 0:2]   = 0.5 * dt**2 * rotation
G[[IDX_VX, IDX_VY], 0:2] = dt * rotation
G[IDX_YAW, 2]            = dt
```

`dynamics.md` §2 writes the injection with a minus sign ($-R\,\mathbf{n}_a$,
$-n_g$), while the code uses a plus sign. Both give the same $Q_k$, because
$Q_k$ is quadratic in $G$ ($(-G)\Sigma(-G)^T = G\Sigma G^T$).

### 3.3 The resulting $Q_k$

With $\Sigma_w = \operatorname{diag}(\sigma_a^2, \sigma_a^2, \sigma_g^2, \sigma_{bw,a}^2, \sigma_{bw,a}^2, \sigma_{bw,g}^2)$,
the product $G_k \Sigma_w G_k^T$ has a simple block structure. The code uses the
same $\sigma_a$ for both accelerometer axes, so the rotation cancels:
$R(\psi)\,\sigma_a^2 I\,R(\psi)^T = \sigma_a^2 I$. **As a result, $Q_k$ does not
depend on yaw.** In state order $(\mathbf{p}, \mathbf{v}, \psi, \mathbf{b}_a, b_g)$:

$$
Q_k =
\begin{bmatrix}
\tfrac14 \sigma_a^2 \Delta t^4\, I_2 & \tfrac12 \sigma_a^2 \Delta t^3\, I_2 & 0 & 0 & 0 \\
\tfrac12 \sigma_a^2 \Delta t^3\, I_2 & \sigma_a^2 \Delta t^2\, I_2 & 0 & 0 & 0 \\
0 & 0 & \sigma_g^2 \Delta t^2 & 0 & 0 \\
0 & 0 & 0 & \sigma_{bw,a}^2 \Delta t\, I_2 & 0 \\
0 & 0 & 0 & 0 & \sigma_{bw,g}^2 \Delta t
\end{bmatrix}.
$$

This matrix is block diagonal. Position and velocity are correlated only with
each other, and yaw and the biases have no cross terms at all. All the
correlation between blocks that $P$ builds up comes from $F_k P F_k^T$.

## 4. Per-sample σ versus a continuous PSD

IMU datasheets usually give noise as a **density**, for example
µg/√Hz or (°/s)/√Hz. That corresponds to $Q_c$ in §2, not to the per-sample
$\sigma$ of §3.2. The two are linked by

$$
q = \sigma^2 \Delta t \quad\Longleftrightarrow\quad \sigma = \frac{\sqrt{q}}{\sqrt{\Delta t}} = \sqrt{q\, f_s} ,
$$

which is the usual band-limited white-noise discretization at sample rate
$f_s = 1/\Delta t$. It is the same `sig0 * randn() / sqrt(dt)` convention as in
`my_test_kf/process_noise.py`. With this link, the position/velocity block of
§3.3 can be compared with the exact integrated-white-noise result
($\Phi(\tau)L = [\tau, 1]^T$ per axis):

| entry | code (ZOH per-sample) | exact white noise, $q = \sigma_a^2\Delta t$ |
|---|---|---|
| $\operatorname{Var}(v)$ | $\sigma_a^2 \Delta t^2$ | $q\,\Delta t = \sigma_a^2 \Delta t^2$ |
| $\operatorname{Cov}(p, v)$ | $\tfrac12 \sigma_a^2 \Delta t^3$ | $\tfrac12 q\,\Delta t^2 = \tfrac12 \sigma_a^2 \Delta t^3$ |
| $\operatorname{Var}(p)$ | $\tfrac14 \sigma_a^2 \Delta t^4$ | $\tfrac13 q\,\Delta t^3 = \tfrac13 \sigma_a^2 \Delta t^4$ |

The velocity and cross terms agree. Position variance differs by a factor of
$\tfrac14$ versus $\tfrac13$. A held value is one constant push. White noise is
spread over the step, and the part that arrives early has longer to integrate
into position. The difference is $O(\Delta t^4)$, so it is small per step
(see §5).

Two practical consequences:

- **Converting a datasheet value.** With a noise density $N$ in (m/s²)/√Hz
  and the filter running at $\Delta t$, use `accel_noise_std` $= N/\sqrt{\Delta t}$.
  The same applies to the gyro. For the bias walks, no conversion is needed:
  they are already densities (§3.1).
- **Changing `dt` changes the noise.** `accel_noise_std` is tied to the sample
  rate. If you change `dt` but keep the same `accel_noise_std`, the equivalent
  PSD $q = \sigma_a^2 \Delta t$ changes too. The bias-walk parameters do not
  have this problem.

## 5. What the approximation drops, with numbers

$G_k \Sigma_w G_k^T$ ignores two effects that the exact $Q_k$ of §2 includes:

1. **Coupling during the step.** Noise that enters one state reaches other
   states through $A$ before the step ends. A bias random-walk increment moves
   velocity through $-R(\psi)$. Gyro noise moves yaw, which rotates the
   acceleration into velocity. The leading terms are
   $\operatorname{Cov}(v_x, b_{ax}) \approx -\cos\psi\, \sigma_{bw,a}^2 \Delta t^2/2$,
   $\operatorname{Cov}(\psi, b_g) \approx -\sigma_{bw,g}^2 \Delta t^2/2$ and
   $\operatorname{Cov}(v_x, \psi) \approx A_{v_x\psi}\, \sigma_g^2 \Delta t^3/2$.
2. **Held versus white noise.** This is the $\tfrac14$ versus $\tfrac13$
   difference in §4.

[`misc/compare_process_noise.py`](../misc/compare_process_noise.py)
evaluates both at $\Delta t = 0.1$ s with the default `EkfConfig`,
$\psi = 0.6$ rad and $\mathbf{u} = [0.3, 0.1, 0.2]$:

```bash
uv run python -m imu_gnss_ekf_training.misc.compare_process_noise
```

| entry | code $Q_k$ | Van Loan $Q_k$ | note |
|---|---|---|---|
| $x, x$ | 3.600e-07 | 4.801e-07 | $\tfrac14$ vs $\tfrac13$ |
| $x, v_x$ | 7.200e-06 | 7.201e-06 | |
| $v_x, v_x$ | 1.440e-04 | 1.440e-04 | |
| $\psi, \psi$ | 2.250e-06 | 2.252e-06 | |
| $b_{ax}, b_{ax}$ | 1.000e-05 | 1.000e-05 | exact |
| $b_g, b_g$ | 6.250e-07 | 6.250e-07 | exact |
| $v_x, b_{ax}$ | 0 | -4.127e-07 | coupling, correlation ≈ 1 % |
| $\psi, b_g$ | 0 | -3.125e-08 | coupling, correlation ≈ 3 % |
| $v_x, \psi$ | 0 | -2.836e-08 | coupling |
| $x, \psi$ | 0 | -9.452e-10 | coupling |

The diagonal terms that matter most (velocity, yaw, biases) agree to within
0.1 %. The missing cross terms are a few percent of a correlation per step, and
$F_k P F_k^T$ rebuilds them anyway on the next step. Their size shrinks as
$\Delta t$ decreases. At typical IMU rates, the uncertainty in how the noise
parameters are tuned is much larger than these differences. This is why the
simple $G \Sigma G^T$ form is the standard choice for this kind of filter.

## 6. Summary

| Source | Code parameter | Unit | $G$ entry | Contribution to $Q_k$ | Exactness |
|---|---|---|---|---|---|
| accel noise | `accel_noise_std` | m/s² per sample | $\tfrac12\Delta t^2 R$, $\Delta t R$ | $\sigma_a^2[\tfrac14\Delta t^4, \tfrac12\Delta t^3; \cdot, \Delta t^2]$ | ZOH model (pos. var. ¼ vs ⅓) |
| gyro noise | `gyro_noise_std` | rad/s per sample | $\Delta t$ | $\sigma_g^2 \Delta t^2$ on $\psi$ | exact for ZOH |
| accel bias walk | `accel_bias_walk_std` | m/s²/√s | $\sqrt{\Delta t}$ | $\sigma_{bw,a}^2 \Delta t$ on $b_{ax}, b_{ay}$ | exact (no coupling) |
| gyro bias walk | `gyro_bias_walk_std` | rad/s/√s | $\sqrt{\Delta t}$ | $\sigma_{bw,g}^2 \Delta t$ on $b_g$ | exact (no coupling) |

`EkfConfig` and `SimulatorConfig` use the same four parameters with the same
meaning. When the EKF is given the simulator's values, $Q_k$ matches the noise
that was actually injected, apart from the small effects listed in §5.
