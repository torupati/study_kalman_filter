# Visual odometry (delta pose) simulation

A visual SLAM or LiDAR SLAM front end tracks the vehicle from one frame to the
next. Its output that is reliable in the short term is the **relative pose
between two frames**: how far the vehicle moved and turned since the last
tracked frame. Its absolute pose, in its own map frame, drifts. The simulator
imitates this front end by computing that relative pose from the ground-truth
trajectory and adding noise.

The implementation is `simulate_visual_odometry` in
[`simulator.py`](../simulator.py), and `simulate_scenario` calls it for every run.

## 1. Measurement definition

The filter is planar, so a pose is $\mathbf{x} = (x, y, \psi)$, with position
$\mathbf{p} = (x, y)$ in the navigation frame and yaw $\psi$ (see
[coordinate frames](coordinate_frames.md)). For two frames $i$ (earlier) and $j$
(later), the VO measurement is the SE(2) pose of frame $j$ seen from frame $i$:

$$
\Delta\mathbf{x}_{ij} =
\begin{bmatrix} \Delta x \\ \Delta y \\ \Delta\psi \end{bmatrix} =
\begin{bmatrix}
R(\psi_i)^\top (\mathbf{p}_j - \mathbf{p}_i) \\
\operatorname{wrap}(\psi_j - \psi_i)
\end{bmatrix},
\qquad
R(\psi) = \begin{bmatrix} \cos\psi & -\sin\psi \\ \sin\psi & \cos\psi \end{bmatrix}
$$

Written out:

$$
\begin{aligned}
\Delta x &= \cos\psi_i\,(x_j - x_i) + \sin\psi_i\,(y_j - y_i) \\
\Delta y &= -\sin\psi_i\,(x_j - x_i) + \cos\psi_i\,(y_j - y_i) \\
\Delta\psi &= \operatorname{wrap}(\psi_j - \psi_i)
\end{aligned}
$$

- $\Delta x$ is the distance moved **forward** and $\Delta y$ the distance moved
  **left**, both along the body axes of frame $i$. They do not depend on where the
  vehicle is or which way it faces in the navigation frame, which is why a camera
  or LiDAR can measure them without knowing its absolute pose.
- $\operatorname{wrap}$ maps to $[-\pi, \pi]$ (`arctan2(sin, cos)`, as in the EKF),
  so a turn from 170° to −175° gives $+15°$, not $-345°$.

The inverse operation, **composition**, applies a delta to a pose:

$$
\mathbf{p}_j = \mathbf{p}_i + R(\psi_i)\,[\Delta x, \Delta y]^\top,
\qquad
\psi_j = \operatorname{wrap}(\psi_i + \Delta\psi)
$$

Chaining compositions from a known start pose is VO dead reckoning. The two
operations are `relative_pose_2d` and `compose_pose_2d` in `simulator.py`.

## 2. Frame timing

VO frames lie on the IMU sample grid, one every
`vo_stride = round(vo_period / dt)` samples (for example 10 fps with 100 Hz IMU gives
a stride of 10). This is the same way GNSS epochs are placed. The EKF then sees each
VO measurement at an IMU step, with no interpolation.

Frame 0 is always tracked. Each later frame loses tracking with probability
`vo_dropout_probability`. When a frame is dropped, the next tracked frame is matched
against the **last tracked frame**, as a real tracker matches against its last
keyframe. The delta then spans two or more strides. For this reason each measurement
stores both of its frame indices:

```
tracked frames:  0   10   20   (30 lost)   40   50
deltas:          0→10  10→20   20→40        40→50
```

With no dropout, every delta spans exactly one stride. Whether or not frames are
dropped, the deltas form an unbroken chain: `vo_from_index[k+1] == vo_to_index[k]`.

## 3. Noise model

VO error grows with motion: larger displacements mean fewer shared features or
points, and more error in feature matching or scan registration. On top of that,
a small per-frame error exists even when standing still. The simulator models the
standard deviation as a floor plus a part proportional to the motion:

$$
\sigma_t = \sigma_{t,0} + k_t\,\lVert(\Delta x, \Delta y)\rVert,
\qquad
\sigma_\psi = \sigma_{\psi,0} + k_\psi\,\lvert\Delta\psi\rvert
$$

The noise is zero-mean Gaussian and independent across axes and frames:

$$
\tilde{\Delta\mathbf{x}} = \Delta\mathbf{x} + \mathbf{n},
\qquad
\mathbf{n} \sim \mathcal{N}\!\left(0,\ \operatorname{diag}(\sigma_t^2, \sigma_t^2, \sigma_\psi^2)\right)
$$

$\sigma_t$ and $\sigma_\psi$ are computed from the **true** delta, so the stored
covariance is exactly the one the noise was drawn from. A real front end has to
estimate its covariance from the noisy measurement instead.

| `SimulatorConfig` field | symbol | default | meaning |
|---|---|---|---|
| `vo_period` | | 0.1 s | frame period (10 fps) |
| `vo_translation_std` | $\sigma_{t,0}$ | 5 mm | translation noise floor per frame |
| `vo_translation_std_per_m` | $k_t$ | 0.01 | extra translation noise per meter moved (1 %) |
| `vo_yaw_std` | $\sigma_{\psi,0}$ | 0.05° | yaw noise floor per frame |
| `vo_yaw_std_per_rad` | $k_\psi$ | 0.01 | extra yaw noise per radian turned (1 %) |
| `vo_dropout_probability` | | 0 | probability that a frame loses tracking |

A 1 % translation drift is in the range of good LiDAR odometry and ordinary visual
odometry. Raise $k_t$ to a few percent to mimic a weak camera setup.

Because the floor applies to every frame, dead-reckoned yaw drifts like a random
walk even when the vehicle is stationary: after $N$ frames the yaw error has
$\sigma \approx \sigma_{\psi,0}\sqrt{N}$, which is about 1.2° after 60 s at 10 fps.
A real tracker that sees a static scene usually drifts less than this. Lower the
floors to model that case.

The VO noise is drawn from the simulator's random generator **after** all IMU and
GNSS draws. Adding or changing VO settings therefore leaves the truth, IMU, and
GNSS outputs for a given `seed` unchanged.

## 4. Worked example: circle scenario at 10 fps

In `circle`, the vehicle speeds up at 0.5 m/s² to $v = 2$ m/s and then drives on a
circle of radius $R = 10$ m, turning at $\omega = v/R = 0.2$ rad/s
(`dt = 0.01`, `vo_period = 0.1`).

At cruise speed, one frame covers an arc of angle
$\Delta\psi = \omega\,\Delta t = 0.02$ rad = 1.146°. In the body frame of the
earlier pose, the chord of that arc is

$$
\Delta x = R\sin\Delta\psi = 0.19999\ \text{m},
\qquad
\Delta y = R\,(1 - \cos\Delta\psi) = 0.0020\ \text{m}
$$

$\Delta y$ is small and positive because the vehicle turns left (counter-clockwise).
Output from `simulate_scenario(SimulatorConfig(scenario="circle", dt=0.01))`, seed 7:

| $t_i$ → $t_j$ [s] | true $(\Delta x, \Delta y, \Delta\psi)$ | measured | $\sigma_t$, $\sigma_\psi$ |
|---|---|---|---|
| 0.0 → 0.1 | 0.0025 m, 0.0000 m, 0.013° | −0.0081 m, 0.0008 m, 0.073° | 5.0 mm, 0.050° |
| 2.0 → 2.1 | 0.1025 m, 0.0005 m, 0.586° | 0.1029 m, 0.0025 m, 0.653° | 6.0 mm, 0.056° |
| 5.0 → 5.1 | 0.2000 m, 0.0018 m, 1.146° | 0.1850 m, 0.0067 m, 1.221° | 7.0 mm, 0.062° |
| 10.0 → 10.1 | 0.2000 m, 0.0018 m, 1.146° | 0.1993 m, −0.0014 m, 1.185° | 7.0 mm, 0.062° |

The true $\Delta y$ at cruise is 0.0018 m rather than 0.0020 m because the
simulator integrates in discrete steps. Each step's velocity is rotated by the yaw at
the start of the step, so the velocity heading lags the yaw by
$\omega\,dt/2 = 0.001$ rad. The chord then makes an angle of
$\Delta\psi/2 - 0.001 = 0.009$ rad with the body x axis, instead of 0.010 rad,
and $\Delta y \approx 0.2 \times 0.009 = 0.0018$ m.

Dead reckoning from the noisy deltas alone (no IMU or GNSS), over 60 s and 600
deltas, with default settings and seed 7:

| scenario | path length | final position error | final yaw error |
|---|---|---|---|
| `stationary` | 0 m | 0.06 m | −1.7° |
| `forward_back` | 10 m | 0.06 m | −1.7° |
| `demo2` | 86 m | 0.45 m | −1.8° |
| `circle` | 116 m | 0.54 m | −2.2° |

Yaw error dominates: it rotates every later displacement, so position error grows
with distance traveled. This drift is what fusion with GNSS, or loop closure, is
meant to correct.

## 5. Output arrays

`simulate_scenario` adds the following entries, where $M$ is the number of deltas
(the number of tracked frames minus 1):

| key | shape | content |
|---|---|---|
| `vo_from_index` | (M,) int | IMU-grid index of frame $i$ (use `time[...]` for its time) |
| `vo_to_index` | (M,) int | IMU-grid index of frame $j$ |
| `vo_delta_truth` | (M, 3) | true $(\Delta x, \Delta y, \Delta\psi)$ |
| `vo_delta_measurements` | (M, 3) | noisy measurement |
| `vo_covariances` | (M, 3, 3) | $\operatorname{diag}(\sigma_t^2, \sigma_t^2, \sigma_\psi^2)$ |

This is the same content as a g2o `EDGE_SE2` (a relative pose and its information
matrix, $\Sigma^{-1}$). It can be exported to pose-graph tools without conversion.

```python
from imu_gnss_ekf_training import SimulatorConfig, simulate_scenario

sim = simulate_scenario(SimulatorConfig(scenario="circle", dt=0.01, vo_period=0.1))
t_meas = sim["time"][sim["vo_to_index"]]
delta = sim["vo_delta_measurements"]
```

## 6. Fusing VO in the EKF (stochastic cloning)

The measurement depends on the pose at **two** times, but the 8-state EKF holds
only the current pose. [`ekf.py`](../ekf.py) solves this with **stochastic
cloning**, as VIO/MSCKF-style filters do. While a VO frame is open, the state is
augmented to 11 elements:

$$
\mathbf{x}_{aug} = [\,x, y, v_x, v_y, \psi, b_{ax}, b_{ay}, b_g,\ x_c, y_c, \psi_c\,]^\top
$$

(`IDX_CLONE_X`, `IDX_CLONE_Y`, `IDX_CLONE_YAW` = 8, 9, 10).

1. **Clone** (`ImuGnssEkf.clone_pose`), at frame $i$: $\mathbf{x}_{aug} = J\mathbf{x}$
   and $P_{aug} = J P J^\top$, where $J = \begin{bmatrix} I_8 \\ S \end{bmatrix}$ and
   $S$ selects $(x, y, \psi)$. The clone starts as an exact copy, fully correlated
   with the pose.
2. **Predict** (`predict`, which accepts the augmented state): the clone does not
   change. Its block of $F$ is the identity and its block of $Q$ is zero, so
   $F P F^\top$ only carries its cross-covariance with the moving state forward. GNSS
   updates at intermediate steps also correct the clone through that correlation.
3. **Update** (`update_vo`), at frame $j$: with $h(\mathbf{x}_{aug}) = \Delta\mathbf{x}(\mathbf{c}, \mathbf{x})$
   from §1 and $R = $ `vo_covariances[k]`, the innovation is $\mathbf{z} - h$ with the
   yaw component wrapped. The update is the same Joseph-form update as for GNSS.
4. **Drop the clone** (`drop_clone`): keep the 8-state block. This is exact
   marginalization for a Gaussian.

With $\mathbf{d} = \mathbf{p} - \mathbf{p}_c$, the nonzero Jacobian blocks of $h$
(`vo_measurement_model`) are:

$$
\frac{\partial h}{\partial (x, y, \psi)} =
\begin{bmatrix} R(\psi_c)^\top & 0 \\ 0 & 1 \end{bmatrix},
\qquad
\frac{\partial h}{\partial (x_c, y_c, \psi_c)} =
\begin{bmatrix} -R(\psi_c)^\top & \dfrac{\partial R(\psi_c)^\top}{\partial \psi_c}\,\mathbf{d} \\ 0 & -1 \end{bmatrix}
$$

$$
\frac{\partial R(\psi_c)^\top}{\partial \psi_c}\,\mathbf{d} =
\begin{bmatrix} -\sin\psi_c\, d_x + \cos\psi_c\, d_y \\ -\cos\psi_c\, d_x - \sin\psi_c\, d_y \end{bmatrix}
= \begin{bmatrix} \Delta y \\ -\Delta x \end{bmatrix}
$$

The tests check these blocks against finite differences.

### Order within one IMU step

`run_filter` processes each step $k$ in this order:

```
predict  →  VO update (if k == vo_to_index[m]; then drop the clone)
         →  GNSS update (if available)
         →  clone the pose (if k == vo_from_index[m+1])
```

The simulator's deltas chain ($i_{m+1} = j_m$), so at most one clone is open at a
time. `run_filter` rejects deltas that overlap or go backward. The histories it
returns stay 8-state, and `vo_innovations` / `vo_innovation_covariances` hold the
VO innovation and $S$ per step, NaN on steps without a VO update.

### Command line

```bash
# simulate VO at 10 fps and fuse it (VO is simulated always, fused only with --use-vo)
uv run python -m imu_gnss_ekf_training.demo --scenario circle --dt 0.01 --gnss-period 1.0 --use-vo

# weaker VO: 3 % drift, 0.2 deg/frame yaw floor, 10 % lost frames
uv run python -m imu_gnss_ekf_training.demo --scenario circle --dt 0.01 --use-vo \
  --vo-translation-std-per-m 0.03 --vo-yaw-std-deg 0.2 --vo-dropout 0.1
```

| flag | `SimulatorConfig` field |
|---|---|
| `--use-vo` | (fuse in the EKF; off by default) |
| `--vo-period` | `vo_period` [s] |
| `--vo-translation-std` | `vo_translation_std` [m] |
| `--vo-translation-std-per-m` | `vo_translation_std_per_m` |
| `--vo-yaw-std-deg` | `vo_yaw_std`, given in degrees |
| `--vo-yaw-std-per-rad` | `vo_yaw_std_per_rad` |
| `--vo-dropout` | `vo_dropout_probability` |

The EKF uses the covariance reported with each delta (`vo_covariances`) as $R$, as
it would with a real front end that reports its own covariance. In simulation
this is the exact noise covariance, which is optimistic compared with a real system.

## 7. Results and a known limitation

Error RMSE over the second half of a 60 s run (IMU 100 Hz, GNSS 1 Hz, VO 10 fps,
default noise, seed 7). "GNSS denied" means the only GNSS fix is at $t = 0$.

| scenario | GNSS | IMU + GNSS: position / yaw | + VO: position / yaw |
|---|---|---|---|
| `circle` | 15 % dropout | 0.92 m / 15.9° | 0.30 m / 1.9° |
| `circle` | denied | 36 m / 52° | 2.9 m / 2.9° |
| `demo2` | 15 % dropout | 0.95 m / 9.7° | 0.44 m / 2.2° |
| `stationary` | 15 % dropout | 1.13 m / 32.9° | 0.11 m / 2.9° |
| `stationary` | denied | 105 m / 52° | 2.1 m / 4.4° |

VO measures velocity (as body-frame displacement) and yaw rate, so it limits the
IMU's quadratic position drift and makes the gyro bias observable even when the
vehicle is standing still. The VO normalized innovation squared
$\nu^\top S^{-1}\nu$ averages 3.08, close to the $\chi^2(3)$ mean of 3, so the VO
updates are consistent with their covariance.

**Limitation: overconfident yaw when GNSS is denied.** IMU and VO measure only
body-frame quantities, and a single GNSS fix constrains position, not heading.
Rotating the whole trajectory about the start point therefore leaves every
measurement unchanged, and absolute yaw is **unobservable**. If the covariance is
propagated with all Jacobians evaluated at the true state, the yaw $\sigma$ stays
at its 45° prior for the whole run, which confirms the Jacobians are correct. The
EKF, however, evaluates them at its changing estimates. That leaks spurious
information into the unobservable direction: in the GNSS-denied circle run, yaw
$\sigma$ shrinks to about 4.5° while the actual yaw error is about 17°. This is the
well-known EKF-VIO inconsistency. The standard fix is **first-estimates Jacobians
(FEJ)**, which evaluate each Jacobian at the first estimate of the states it
involves, so the linearized system keeps the unobservable direction. The strict
`xfail` test `test_yaw_uncertainty_stays_near_prior_without_gnss` records the
current behaviour. With regular GNSS, yaw becomes observable whenever the vehicle
accelerates, and this problem does not appear.

## 8. VO-only navigation: the estimate lives in a rotated frame

The simulator can make a data set with every sensor (IMU, GNSS, VO) while the EKF
navigates with **IMU + VO only**. Without GNSS, nothing tells the filter where the
vehicle starts in the world or which way it faces. Like a SLAM system, it defines
its own **navigation frame** by the start pose: the EKF starts at $(0, 0)$ with
heading $0$. The vehicle, however, starts at a world pose
$(\mathbf{p}_0, \psi_0)$ (`SimulatorConfig.initial_position`, `initial_yaw`,
`--initial-pose X Y YAW_DEG`). The estimated trajectory is therefore the true one
expressed in the start frame:

$$
\mathbf{p}^{world} = R(\psi_0)\,\mathbf{p}^{nav} + \mathbf{p}_0,
\qquad
\psi^{world} = \psi^{nav} + \psi_0
$$

It is **rotated by $\psi_0$ and shifted by $\mathbf{p}_0$**, plus whatever drift the
navigation accumulated. The IMU and VO measurements do not depend on the start
pose: the simulator produces identical IMU data and VO deltas for any start pose,
and the tests check this. Only GNSS sees the world frame.

### Anchoring the EKF

With `--no-gnss-update`, `demo.py` gives the EKF a tight initial pose prior
(0.01 m, 0.1°) instead of the usual 5 m / 45°. In its own frame, the start pose is
exact by definition. A loose prior would put uncertainty in exactly the
unobservable directions (global position and heading), and the estimate-linearized
EKF would then leak spurious information into them (§7), which distorts even the
relative trajectory. Override the prior with `--ekf-initial-position-std` and
`--ekf-initial-yaw-std-deg`.

### Checking the rotation

`alignment.py` compares the nav-frame estimate with the world-frame truth in two
ways:

- **Mapped with the true start pose** (`anchor`): apply the transform above with the
  true $(\mathbf{p}_0, \psi_0)$. What remains is the drift of the navigation itself.
- **Best-fit rigid transform** (`fitted`): the least-squares rotation and translation
  from estimate to truth (2D Kabsch/Umeyama, no scale). This is the "aligned ATE"
  used to evaluate SLAM, which needs no knowledge of the start pose. The fitted
  rotation recovers $\psi_0$, except that it also absorbs part of the heading drift.

```bash
# GNSS and VO both simulated; the EKF uses IMU + VO only. The vehicle starts at (30, -20) heading 40°.
uv run python -m imu_gnss_ekf_training.demo --scenario demo2 --dt 0.01 \
  --use-vo --no-gnss-update --initial-pose 30 -20 40

# the same data set as CSV (imu/gnss/vo_observations.csv, true_state.csv)
uv run python -m imu_gnss_ekf_training.record_sensors demo2 --no-show --save-true-state --initial-pose 30 -20 40
```

The demo writes `ekf_frame_alignment.png`. It has three panels: the estimate in its
own frame over the world-frame truth, the estimate mapped with the true start pose,
and the position error over time. It also prints:

| | rotation | translation | position RMSE |
|---|---|---|---|
| true start pose | 40.00° | (30.00, −20.00) m | 0.32 m after mapping (yaw RMSE 1.2°) |
| best fit | 40.72° | (30.13, −20.24) m | 0.09 m |
| no alignment | | | ≈ 30 m (the printed `position_rmse_m`) |

(`demo2`, 60 s, 10 fps VO with default noise, seed 7.) The best fit has a lower RMSE
than the true start pose because it is free to rotate the whole trajectory to absorb
the heading drift. In `circle`, the best-fit rotation is 41.3° against the true 40°.
The circle's shape looks the same after rotation, but its position relative to the
start still reveals the rotation.

[vo_only_navigation.md](vo_only_navigation.md) looks at this run in detail: how the error grows, a VO noise sweep, and why the IMU adds little without a known gyro bias.

The `vo_observations.csv` columns are `time_from_s, time_to_s, dx_body_m, dy_body_m,
dyaw_rad, std_dx_m, std_dy_m, std_dyaw_rad`, one row per delta (§5).
