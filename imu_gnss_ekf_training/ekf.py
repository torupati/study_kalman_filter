from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STATE_SIZE = 8
MEAS_SIZE = 4
IDX_X = 0
IDX_Y = 1
IDX_VX = 2
IDX_VY = 3
IDX_YAW = 4
IDX_BAX = 5
IDX_BAY = 6
IDX_BG = 7


@dataclass(frozen=True)
class EkfConfig:
    accel_noise_std: float = 0.12
    gyro_noise_std: float = 0.015
    accel_bias_walk_std: float = 0.01
    gyro_bias_walk_std: float = 0.0025
    gnss_position_std: float = 1.5
    gnss_velocity_std: float = 0.35


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap an angle to [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class ImuGnssEkf:
    """Extended Kalman filter for planar inertial/GNSS fusion."""

    def __init__(self, config: EkfConfig):
        self.config = config
        self.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        self.R = np.diag(
            [
                config.gnss_position_std**2,
                config.gnss_position_std**2,
                config.gnss_velocity_std**2,
                config.gnss_velocity_std**2,
            ]
        )

    @staticmethod
    def rotation_matrix(yaw: float) -> np.ndarray:
        c = np.cos(yaw)
        s = np.sin(yaw)
        return np.array([[c, -s], [s, c]], dtype=float)

    def predict(self, state: np.ndarray, covariance: np.ndarray, imu_sample: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Propagate the state using IMU measurements as control inputs."""
        ax_meas, ay_meas, omega_meas = imu_sample

        yaw = state[IDX_YAW]
        bax = state[IDX_BAX]
        bay = state[IDX_BAY]
        bg = state[IDX_BG]

        accel_body = np.array([ax_meas - bax, ay_meas - bay], dtype=float)
        rotation = self.rotation_matrix(yaw)
        accel_nav = rotation @ accel_body

        predicted = state.copy()
        predicted[IDX_X] += state[IDX_VX] * dt + 0.5 * accel_nav[0] * dt**2
        predicted[IDX_Y] += state[IDX_VY] * dt + 0.5 * accel_nav[1] * dt**2
        predicted[IDX_VX] += accel_nav[0] * dt
        predicted[IDX_VY] += accel_nav[1] * dt
        predicted[IDX_YAW] = wrap_angle(yaw + (omega_meas - bg) * dt)

        c = np.cos(yaw)
        s = np.sin(yaw)
        dax_dyaw = -s * accel_body[0] - c * accel_body[1]
        day_dyaw = c * accel_body[0] - s * accel_body[1]

        F = np.eye(STATE_SIZE)
        F[IDX_X, IDX_VX] = dt
        F[IDX_Y, IDX_VY] = dt
        F[IDX_X, IDX_YAW] = 0.5 * dt**2 * dax_dyaw
        F[IDX_Y, IDX_YAW] = 0.5 * dt**2 * day_dyaw
        F[IDX_VX, IDX_YAW] = dt * dax_dyaw
        F[IDX_VY, IDX_YAW] = dt * day_dyaw
        F[IDX_X, IDX_BAX] = -0.5 * dt**2 * c
        F[IDX_X, IDX_BAY] = -0.5 * dt**2 * s
        F[IDX_Y, IDX_BAX] = -0.5 * dt**2 * s
        F[IDX_Y, IDX_BAY] = -0.5 * dt**2 * c
        F[IDX_VX, IDX_BAX] = -dt * c
        F[IDX_VX, IDX_BAY] = -dt * s
        F[IDX_VY, IDX_BAX] = -dt * s
        F[IDX_VY, IDX_BAY] = -dt * c
        F[IDX_YAW, IDX_BG] = -dt

        G = np.zeros((STATE_SIZE, 6), dtype=float)
        G[IDX_X:IDX_Y + 1, 0:2] = 0.5 * dt**2 * rotation
        G[IDX_VX:IDX_VY + 1, 0:2] = dt * rotation
        G[IDX_YAW, 2] = dt
        G[IDX_BAX, 3] = np.sqrt(dt)
        G[IDX_BAY, 4] = np.sqrt(dt)
        G[IDX_BG, 5] = np.sqrt(dt)

        process_noise = np.diag(
            [
                self.config.accel_noise_std**2,
                self.config.accel_noise_std**2,
                self.config.gyro_noise_std**2,
                self.config.accel_bias_walk_std**2,
                self.config.accel_bias_walk_std**2,
                self.config.gyro_bias_walk_std**2,
            ]
        )
        Q = G @ process_noise @ G.T
        predicted_covariance = F @ covariance @ F.T + Q
        predicted_covariance = 0.5 * (predicted_covariance + predicted_covariance.T)
        return predicted, predicted_covariance

    def update(self, predicted_state: np.ndarray, predicted_covariance: np.ndarray, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply a GNSS position/velocity update."""
        innovation = measurement - self.H @ predicted_state
        innovation_covariance = self.H @ predicted_covariance @ self.H.T + self.R
        kalman_gain = predicted_covariance @ self.H.T @ np.linalg.inv(innovation_covariance)

        updated_state = predicted_state + kalman_gain @ innovation
        updated_state[IDX_YAW] = wrap_angle(updated_state[IDX_YAW])

        identity = np.eye(STATE_SIZE)
        residual_projector = identity - kalman_gain @ self.H
        updated_covariance = (
            residual_projector @ predicted_covariance @ residual_projector.T
            + kalman_gain @ self.R @ kalman_gain.T
        )
        updated_covariance = 0.5 * (updated_covariance + updated_covariance.T)
        return updated_state, updated_covariance


def run_filter(
    imu_measurements: np.ndarray,
    gnss_measurements: np.ndarray,
    gnss_available: np.ndarray,
    dt: float,
    config: EkfConfig | None = None,
    initial_state: np.ndarray | None = None,
    initial_covariance: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Run the EKF over a full simulated sequence."""
    if config is None:
        config = EkfConfig()

    ekf = ImuGnssEkf(config)
    state = np.zeros(STATE_SIZE, dtype=float) if initial_state is None else initial_state.astype(float).copy()
    covariance = np.diag([25.0, 25.0, 4.0, 4.0, np.deg2rad(45.0) ** 2, 0.5, 0.5, 0.05])
    if initial_covariance is not None:
        covariance = initial_covariance.astype(float).copy()

    state_history = np.zeros((imu_measurements.shape[0], STATE_SIZE), dtype=float)
    covariance_history = np.zeros((imu_measurements.shape[0], STATE_SIZE, STATE_SIZE), dtype=float)

    if gnss_available[0]:
        state, covariance = ekf.update(state, covariance, gnss_measurements[0])
    state_history[0] = state
    covariance_history[0] = covariance

    for index in range(1, imu_measurements.shape[0]):
        state, covariance = ekf.predict(state, covariance, imu_measurements[index - 1], dt)
        if gnss_available[index]:
            state, covariance = ekf.update(state, covariance, gnss_measurements[index])
        state_history[index] = state
        covariance_history[index] = covariance

    return {
        "state_estimates": state_history,
        "covariances": covariance_history,
    }
