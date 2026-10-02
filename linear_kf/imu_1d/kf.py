"""1D position/velocity Kalman filter with the accelerometer as a control input.

State ``x = [p, v]``. Each IMU sample drives the prediction,

    x_{k+1} = F x_k + B a_meas_k,   F = [[1, dt], [0, 1]],   B = [dt^2/2, dt],

and the accelerometer white noise enters the same way as the input, so
``Q = sigma_a^2 B B^T`` with ``sigma_a`` the per-sample noise std. Position
measurements (``H = [1, 0]``) correct the state when they are available.
"""

from dataclasses import dataclass

import numpy as np

H = np.array([[1.0, 0.0]])


def transition(dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (F, B) for one zero-order-hold step of length dt."""
    F = np.array([[1.0, dt], [0.0, 1.0]])
    B = np.array([0.5 * dt * dt, dt])
    return F, B


class KalmanFilterImu1d:
    def __init__(self, accel_noise_std: float, pos_noise_std: float):
        self.accel_noise_std = accel_noise_std
        self.R = np.array([[pos_noise_std**2]])

    def Q(self, dt: float) -> np.ndarray:
        _, B = transition(dt)
        return self.accel_noise_std**2 * np.outer(B, B)

    def predict(self, x: np.ndarray, P: np.ndarray, acc_meas: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
        F, B = transition(dt)
        x = F @ x + B * acc_meas
        P = F @ P @ F.T + self.Q(dt)
        return x, 0.5 * (P + P.T)

    def update(self, x: np.ndarray, P: np.ndarray, pos_meas: float) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Position update. Returns (x, P, innovation, innovation variance S)."""
        S = H @ P @ H.T + self.R
        K = np.linalg.solve(S, H @ P).T  # P H^T S^-1 (S and P are symmetric)
        e = pos_meas - (H @ x)[0]
        x = x + K[:, 0] * e
        I_KH = np.eye(2) - K @ H
        P = I_KH @ P @ I_KH.T + K @ self.R @ K.T  # Joseph form
        return x, 0.5 * (P + P.T), e, S[0, 0]


@dataclass
class FilterResult:
    x: np.ndarray  # (N, 2) posterior state at each IMU time
    P: np.ndarray  # (N, 2, 2) posterior covariance
    innovation: np.ndarray  # (N,) NaN where no position update
    S: np.ndarray  # (N,) innovation variance, NaN where no position update

    @property
    def nis(self) -> np.ndarray:
        """Normalized innovation squared e^2 / S (NaN where no update)."""
        return self.innovation**2 / self.S


def run_filter(
    kf: KalmanFilterImu1d,
    t: np.ndarray,
    acc_meas: np.ndarray,
    pos_meas: np.ndarray,
    x0: np.ndarray,
    P0: np.ndarray,
) -> FilterResult:
    """Run update-then-predict over IMU samples; pos_meas is NaN where absent.

    (x0, P0) is the prior at t[0]. acc_meas[k] drives the step t[k] -> t[k+1].
    """
    n = len(t)
    xs = np.empty((n, 2))
    Ps = np.empty((n, 2, 2))
    innov = np.full(n, np.nan)
    S = np.full(n, np.nan)
    x, P = np.asarray(x0, dtype=float), np.asarray(P0, dtype=float)
    for k in range(n):
        if not np.isnan(pos_meas[k]):
            x, P, innov[k], S[k] = kf.update(x, P, pos_meas[k])
        xs[k], Ps[k] = x, P
        if k < n - 1:
            x, P = kf.predict(x, P, acc_meas[k], t[k + 1] - t[k])
    return FilterResult(xs, Ps, innov, S)


def run_position_only_filter(
    t: np.ndarray,
    pos_meas: np.ndarray,
    accel_psd_std: float,
    pos_noise_std: float,
    x0: np.ndarray,
    P0: np.ndarray,
) -> FilterResult:
    """Baseline without IMU: the random-acceleration model of doc/kf_basics.md.

    Acceleration is treated as white noise with density ``accel_psd_std``, so
    ``Q = q [[dt^3/3, dt^2/2], [dt^2/2, dt]]`` and the prediction uses no input.
    Steps are taken at every sample of ``t`` so the output aligns with ``run_filter``.
    """
    q = accel_psd_std**2
    R = np.array([[pos_noise_std**2]])
    n = len(t)
    xs = np.empty((n, 2))
    Ps = np.empty((n, 2, 2))
    innov = np.full(n, np.nan)
    S = np.full(n, np.nan)
    x, P = np.asarray(x0, dtype=float), np.asarray(P0, dtype=float)
    for k in range(n):
        if not np.isnan(pos_meas[k]):
            Sk = H @ P @ H.T + R
            K = np.linalg.solve(Sk, H @ P).T
            innov[k] = pos_meas[k] - x[0]
            S[k] = Sk[0, 0]
            x = x + K[:, 0] * innov[k]
            I_KH = np.eye(2) - K @ H
            P = I_KH @ P @ I_KH.T + K @ R @ K.T
        xs[k], Ps[k] = x, P
        if k < n - 1:
            dt = t[k + 1] - t[k]
            F, _ = transition(dt)
            Q = q * np.array([[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]])
            x = F @ x
            P = F @ P @ F.T + Q
    return FilterResult(xs, Ps, innov, S)
