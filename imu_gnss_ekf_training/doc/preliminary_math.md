# Preliminary mathematics for Kalman filtering

This note collects the probability/linear-algebra background used in
[`dynamics.md`](dynamics.md) and in the EKF implementation (`ekf.py`): variance
and covariance, how covariance transforms under linear maps, white noise, the
Schur complement, and why a Gaussian prior updated with a Gaussian likelihood
stays Gaussian. Together these are exactly what justify propagating just a
mean $\hat{\mathbf{x}}$ and covariance $P$ instead of a full probability
distribution.

## 1. Variance and covariance

For a scalar random variable $X$ with mean $\mu = E[X]$, the variance is

$$
\operatorname{Var}(X) = E[(X-\mu)^2] = E[X^2] - \mu^2 .
$$

For two scalar random variables $X, Y$, the covariance is

$$
\operatorname{Cov}(X,Y) = E[(X-\mu_X)(Y-\mu_Y)] .
$$

For a random **vector** $\mathbf{x} \in \mathbb{R}^n$ with mean
$\boldsymbol{\mu} = E[\mathbf{x}]$, the covariance *matrix* collects all
pairwise (co)variances of its components:

$$
P = \operatorname{Cov}(\mathbf{x}) = E\big[(\mathbf{x}-\boldsymbol{\mu})(\mathbf{x}-\boldsymbol{\mu})^T\big],
\qquad P_{ij} = \operatorname{Cov}(x_i, x_j).
$$

$P$ is symmetric ($P = P^T$) and positive semi-definite
($\mathbf{v}^T P \mathbf{v} = \operatorname{Var}(\mathbf{v}^T\mathbf{x}) \ge 0$
for any $\mathbf{v}$), since it's a variance in disguise. This is exactly the
`P` matrix in `ekf.py`, and the `0.5 * (P + P.T)` symmetrization mentioned in
[`dynamics.md`](dynamics.md#4-error-state-model) exists to restore this
property after roundoff error erodes it.

## 2. Linear transformation of covariance

If $\mathbf{y} = A\mathbf{x} + \mathbf{b}$ for a constant matrix $A$ and vector
$\mathbf{b}$, then

$$
E[\mathbf{y}] = A E[\mathbf{x}] + \mathbf{b}, \qquad
\operatorname{Cov}(\mathbf{y}) = A\, \operatorname{Cov}(\mathbf{x})\, A^T .
$$

*Derivation:* let $\boldsymbol{\mu} = E[\mathbf{x}]$, $P = \operatorname{Cov}(\mathbf{x})$.
Then $\mathbf{y} - E[\mathbf{y}] = A(\mathbf{x}-\boldsymbol{\mu})$, so

$$
\operatorname{Cov}(\mathbf{y})
= E\big[A(\mathbf{x}-\boldsymbol{\mu})(\mathbf{x}-\boldsymbol{\mu})^T A^T\big]
= A\, E\big[(\mathbf{x}-\boldsymbol{\mu})(\mathbf{x}-\boldsymbol{\mu})^T\big]\, A^T
= A P A^T .
$$

This one identity is the covariance half of every predict/update step in this
repo: `predicted_covariance = F @ covariance @ F.T + Q` in `ekf.py` is exactly
$P_{k+1} = F_k P_k F_k^T + Q_k$ from
[`dynamics.md` §4](dynamics.md#4-error-state-model), where the $+Q_k$ term
accounts for the independent noise added on top of the linear map (see §3–§4
below for where $Q$ itself comes from).

## 3. White noise

A continuous-time (vector) white noise process $\mathbf{n}(t)$ is defined by

$$
E[\mathbf{n}(t)] = \mathbf{0}, \qquad
E[\mathbf{n}(t)\,\mathbf{n}(t')^T] = Q_c\, \delta(t-t'),
$$

i.e. it's zero-mean, and its autocorrelation is an impulse at $t=t'$: the
noise at any two distinct instants is uncorrelated, however close together
those instants are. $Q_c$ is the noise's **power spectral density**
(intensity), not a variance in the usual sense — $\delta(t-t')$ has units of
$1/\text{time}$, so $Q_c$ has units of $\text{variance}/\text{time}$. This is
why the noises in [`dynamics.md` §2](dynamics.md#2-continuous-time-nonlinear-dynamics)
(e.g. $n_{ax}$, $n_{bax}$) are specified with a $\delta(t-t')$ autocorrelation
rather than a plain variance.

Because white noise isn't a well-defined instantaneous *value* (only its
integral over a time step is), discretizing it over a step $dt$ turns the
density $Q_c$ into a finite increment covariance $Q_c\, dt$ — a $\sqrt{dt}$
factor on the noise itself. This is exactly why
[`dynamics.md` §6](dynamics.md#6-process-noise-jacobian-g-and-q) scales the
bias rows of $G$ by $\sqrt{dt}$: it converts the continuous-time random-walk
density (`*_walk_std`, a $\sigma/\sqrt{\text{s}}$ quantity) into the correct
per-step discrete covariance contribution.

## 4. Block matrices and the Schur complement

Consider a symmetric block matrix

$$
M = \begin{bmatrix} A & B \\ B^T & D \end{bmatrix},
$$

with $A$, $D$ square and invertible. The **Schur complement of $D$ in $M$**
is

$$
M/D \;=\; A - B D^{-1} B^T .
$$

It appears when eliminating one block of variables from a linear system or
when block-inverting $M$:

$$
M^{-1} =
\begin{bmatrix}
(M/D)^{-1} & -(M/D)^{-1} B D^{-1} \\
-D^{-1} B^T (M/D)^{-1} & D^{-1} + D^{-1} B^T (M/D)^{-1} B D^{-1}
\end{bmatrix}.
$$

On its own this is just linear algebra, but §6 below shows it is *precisely*
the object that appears when conditioning a jointly Gaussian random vector —
which is the operation a Kalman update performs.

## 5. Multivariate normal (Gaussian) distribution

A random vector $\mathbf{x} \in \mathbb{R}^n$ is (multivariate) Gaussian,
written $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, P)$, if its density is

$$
p(\mathbf{x}) = \frac{1}{(2\pi)^{n/2} |P|^{1/2}}
\exp\!\Big(-\tfrac12 (\mathbf{x}-\boldsymbol{\mu})^T P^{-1} (\mathbf{x}-\boldsymbol{\mu})\Big),
$$

fully characterized by its mean $\boldsymbol{\mu}$ and covariance $P$ (no
higher moments needed). The key closure property, following directly from §2:

> **Affine maps preserve Gaussianity.** If $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, P)$
> and $\mathbf{y} = A\mathbf{x} + \mathbf{b}$, then
> $\mathbf{y} \sim \mathcal{N}(A\boldsymbol{\mu}+\mathbf{b},\; A P A^T)$.

This is why, *for a linear system*, propagating $(\hat{\mathbf{x}}, P)$ through
$F$ and adding independent Gaussian process noise (§2–§3) exactly tracks the
true distribution of the state — no approximation needed. (For the EKF's
nonlinear $f$, this only holds approximately, which is exactly the
linearization argument in
[`dynamics.md` §4](dynamics.md#4-error-state-model).)

## 6. Joint, conditional, and posterior Gaussians

Let $\mathbf{x}$ and $\mathbf{z}$ be jointly Gaussian:

$$
\begin{bmatrix} \mathbf{x} \\ \mathbf{z} \end{bmatrix} \sim
\mathcal{N}\!\left(
\begin{bmatrix} \boldsymbol{\mu}_x \\ \boldsymbol{\mu}_z \end{bmatrix},\;
\begin{bmatrix} P_{xx} & P_{xz} \\ P_{xz}^T & P_{zz} \end{bmatrix}
\right).
$$

The conditional distribution of $\mathbf{x}$ given an observed value of
$\mathbf{z}$ is **also Gaussian**,
$\mathbf{x}\mid\mathbf{z} \sim \mathcal{N}(\boldsymbol{\mu}_{x|z}, P_{x|z})$,
with

$$
\boldsymbol{\mu}_{x|z} = \boldsymbol{\mu}_x + P_{xz} P_{zz}^{-1}(\mathbf{z}-\boldsymbol{\mu}_z),
\qquad
P_{x|z} = P_{xx} - P_{xz} P_{zz}^{-1} P_{xz}^T .
$$

The conditional covariance $P_{x|z}$ is exactly the **Schur complement of
$P_{zz}$** in the joint covariance matrix (§4) — this is the direct link
between §4 and Gaussian conditioning, and the reason the Schur complement
shows up throughout Kalman-filter derivations.

**Application — the Kalman/EKF update is Bayesian conditioning.** Take a
Gaussian prior $\mathbf{x} \sim \mathcal{N}(\hat{\mathbf{x}}, P)$ and a linear
Gaussian measurement model $\mathbf{z} = H\mathbf{x} + \mathbf{v}$, with
measurement noise $\mathbf{v} \sim \mathcal{N}(\mathbf{0}, R)$ independent of
$\mathbf{x}$. Then $(\mathbf{x}, \mathbf{z})$ are jointly Gaussian with

$$
\boldsymbol{\mu}_z = H\hat{\mathbf{x}}, \qquad
P_{xz} = P H^T, \qquad
P_{zz} = H P H^T + R,
$$

(the last two follow from §2 applied to $\mathbf{z} - \boldsymbol{\mu}_z = H(\mathbf{x}-\hat{\mathbf{x}}) + \mathbf{v}$).
Substituting into the conditional-Gaussian formulas above and writing
$K = P_{xz} P_{zz}^{-1} = P H^T (H P H^T + R)^{-1}$ (the **Kalman gain**) gives

$$
\hat{\mathbf{x}}_{\text{post}} = \hat{\mathbf{x}} + K(\mathbf{z} - H\hat{\mathbf{x}}),
\qquad
P_{\text{post}} = P - K H P = (I - KH) P,
$$

which are exactly the Kalman/EKF measurement-update equations implemented in
`ImuGnssEkf.update` (`ekf.py`) — the innovation $\mathbf{z}-H\hat{\mathbf{x}}$,
gain $K$, and covariance decrease are not a heuristic, they are the closed-form
conditional mean/covariance of a jointly Gaussian pair. (`ekf.py` uses the
algebraically-equivalent Joseph form
$P_{\text{post}} = (I-KH)P(I-KH)^T + KRK^T$ for the covariance update, which is
less prone to losing positive semi-definiteness to roundoff than
$(I-KH)P$, but both are the same $P_{x|z}$ Schur complement in exact
arithmetic.)

Because the *predict* step is an affine map plus independent Gaussian noise
(§2–§3, exact for a linear $f$) and the *update* step is exact Gaussian
conditioning (this section), a **linear**-Gaussian Kalman filter's
$(\hat{\mathbf{x}}, P)$ are the *exact* posterior mean and covariance at every
step — not merely a convenient summary. The EKF in this repo only departs from
that guarantee where $f$ or $h$ is nonlinear (the $\psi$-dependence discussed
in [`dynamics.md` §4–§5](dynamics.md#4-error-state-model)), which is why it's
called *extended*: it applies the same update algebra to a first-order
(Jacobian) linearization instead of an exact linear model.
