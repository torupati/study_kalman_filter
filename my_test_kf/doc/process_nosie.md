# Random acceleration / random jerk process noise

This documents the noise models simulated in [`process_noise.py`](../process_noise.py) (random
acceleration, 2-state) and [`process_noise2.py`](../process_noise2.py) (random jerk, 3-state), and
the closed-form second-moment formulas that each script overlays on its Monte-Carlo variance/covariance
plots to confirm the simulation is correct.

Both scripts drive a chain of integrators with continuous-time white noise of power spectral density
(PSD) $q = \sigma_0^2$ (`sig0` in the code) and compare $N$ independent sample paths against the
theoretical moments below. The match between simulated and analytic curves in
`vel_sample_var.png` / `model2_cov_sim.png` is the "confirmed by plot matching" referred to here.

## Discretization convention

Both scripts sample a continuous white-noise process $w(t)$ with $E[w(t)w(s)] = q\,\delta(t-s)$ at
timestep `dt = 1/Fs`. There are two equivalent ways this shows up in the code:

- `process_noise.py`: `_acc = sig0 * randn() / sqrt(dt)` samples $w(t)$ itself (variance $q/dt$, the
  usual band-limited-white-noise discretization), then integrates with `v += _acc * dt`.
- `process_noise2.py`: `_jerk = sig0 * randn() * sqrt(dt)` directly samples the *increment*
  $dW \sim \mathcal{N}(0, q\,dt)$ and adds it straight into the state (`a += _jerk`).

These are the same thing: `_acc * dt = sig0*randn()*sqrt(dt)`, identical in distribution to `_jerk`
above. Both are Euler–Maruyama integration of a driving Wiener process with PSD $q=\sigma_0^2$.

## General result: integrated white noise

For a chain of integrators driven by white noise at the top, with continuous-time state matrix $A$
(shift/nilpotent, i.e. $\dot x = Ax + Gw$, $x(0)=0$) the covariance of $x(t)$ is the classical
Van Loan integral

$$
Q(t) = q \int_0^t \Phi(\tau)\,GG^T\,\Phi(\tau)^T\,d\tau, \qquad \Phi(\tau) = e^{A\tau}.
$$

Both models below are special cases of this with a single noise input $G$ at the highest derivative.

## Model 1 — random acceleration (`process_noise.py`)

State $[p, v]$, acceleration is white noise:

$$
\dot v = w(t), \qquad \dot p = v, \qquad E[w(t)w(s)] = \sigma_0^2\,\delta(t-s).
$$

Here $A=\begin{bmatrix}0&1\\0&0\end{bmatrix}$, $G=\begin{bmatrix}0\\1\end{bmatrix}$, so
$\Phi(\tau)G = [\tau, 1]^T$ and

$$
Q(t) = \sigma_0^2\int_0^t \begin{bmatrix}\tau^2 & \tau\\ \tau & 1\end{bmatrix} d\tau
= \sigma_0^2\begin{bmatrix} t^3/3 & t^2/2 \\ t^2/2 & t \end{bmatrix}.
$$

i.e.

$$
\mathrm{Var}[v(t)] = \sigma_0^2 t, \qquad
\mathrm{Var}[p(t)] = \tfrac{1}{3}\sigma_0^2 t^3, \qquad
E[p(t)v(t)] = \mathrm{Cov}[p(t),v(t)] = \tfrac{1}{2}\sigma_0^2 t^2.
$$

These are exactly the three reference curves plotted in `vel_sample_var.png`
(`sig0**2 * t`, `(1/3)*sig0**2*t**3`, `(1/2)*sig0**2*t**2`).

## Model 2 — random jerk (`process_noise2.py`)

State $[p, v, a]$, jerk is white noise:

$$
\dot a = w(t), \qquad \dot v = a, \qquad \dot p = v, \qquad E[w(t)w(s)] = \sigma_0^2\,\delta(t-s).
$$

Here $A=\begin{bmatrix}0&1&0\\0&0&1\\0&0&0\end{bmatrix}$, $G=\begin{bmatrix}0\\0\\1\end{bmatrix}$,
$\Phi(\tau) = \begin{bmatrix}1&\tau&\tau^2/2\\0&1&\tau\\0&0&1\end{bmatrix}$, so
$\Phi(\tau)G = [\tau^2/2,\ \tau,\ 1]^T$ and

$$
Q(t) = \sigma_0^2\int_0^t
\begin{bmatrix}
\tau^4/4 & \tau^3/2 & \tau^2/2 \\
\tau^3/2 & \tau^2    & \tau     \\
\tau^2/2 & \tau      & 1
\end{bmatrix} d\tau
= \sigma_0^2
\begin{bmatrix}
t^5/20 & t^4/8 & t^3/6 \\
t^4/8  & t^3/3 & t^2/2 \\
t^3/6  & t^2/2 & t
\end{bmatrix},
$$

ordered as $(p, v, a)$. Written out:

$$
\mathrm{Var}[a(t)] = \sigma_0^2 t, \qquad
\mathrm{Var}[v(t)] = \tfrac{1}{3}\sigma_0^2 t^3, \qquad
\mathrm{Var}[p(t)] = \tfrac{1}{20}\sigma_0^2 t^5,
$$

$$
E[a(t)v(t)] = \mathrm{Cov}[a(t),v(t)] = \tfrac{1}{2}\sigma_0^2 t^2, \qquad
E[a(t)p(t)] = \mathrm{Cov}[a(t),p(t)] = \tfrac{1}{6}\sigma_0^2 t^3,
$$

$$
E[p(t)v(t)] = \mathrm{Cov}[p(t),v(t)] = \tfrac{1}{8}\sigma_0^2 t^4.
$$

These six formulas are exactly the reference curves overlaid in `model2_cov_sim.png`
(`sig0**2*t`, `(1/3)*sig0**2*t**3`, `(1/20)*sig0**2*t**5`, `(1/2)*sig0**2*t**2`,
`(1/8)*sig0**2*t**4`, `(1/6)*sig0**2*t**3`), confirmed there by Monte-Carlo sample covariance
over `N=500` simulated paths matching the analytic curves.

## Relation between the two models

Model 2's $(v,a)$ block and Model 1's $(p,v)$ block have the *same* form
($t^3/3$, $t^2/2$, $t$) shifted one derivative down — expected, since Model 2 is just Model 1's
integrator chain with one extra integration stage prepended (jerk instead of acceleration as the
white-noise input). Model 1's $Q(t)$ is Model 2's bottom-right $2\times2$ block.
