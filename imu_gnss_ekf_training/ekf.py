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

# Stochastic cloning for visual odometry: while a VO frame is open, a copy of the
# pose at that frame is appended to the state as [x_c, y_c, yaw_c].
CLONE_SIZE = 3
AUGMENTED_STATE_SIZE = STATE_SIZE + CLONE_SIZE
IDX_CLONE_X = 8
IDX_CLONE_Y = 9
IDX_CLONE_YAW = 10
VO_MEAS_SIZE = 3


def wrap_angle(angle):
    """Wrap to [-pi, pi] (arctan2 form, as used for yaw throughout)."""
    return np.arctan2(np.sin(angle), np.cos(angle))


@dataclass(frozen=True)
class EkfConfig:
    """EKF tuning parameters.

    The bias walk values are random-walk noise densities, so the per-step
    increment standard deviation is `walk_std * sqrt(dt)`.
    """

    accel_noise_std: float = 0.12
    gyro_noise_std: float = 0.015
    accel_bias_walk_std: float = 0.01
    gyro_bias_walk_std: float = 0.0025
    gnss_position_std: float = 1.5
    gnss_velocity_std: float = 0.35
    initial_position_std: float = 5.0
    initial_velocity_std: float = 2.0
    initial_yaw_std_rad: float = np.deg2rad(45.0)
    initial_accel_bias_std: float = 0.5
    initial_gyro_bias_std: float = 0.05


class ImuGnssEkf:
    """Extended Kalman filter for planar inertial/GNSS fusion, with optional visual odometry.

    `predict`/`update` accept either the 8-element state or the 11-element
    augmented state that carries a pose clone (see `clone_pose`/`update_vo`).
    The clone is constant during predict, so only its cross-covariance with the
    moving state changes.
    """

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
        size = state.shape[0]
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

        # With accel_nav = [c * ax_c - s * ay_c, s * ax_c + c * ay_c]^T and
        # ax_c = ax_meas - bax, ay_c = ay_meas - bay, the bias partials are:
        # d(accel_nav)/d(bax) = [-c, -s]^T and d(accel_nav)/d(bay) = [s, -c]^T.
        # Sized to the (possibly augmented) state: clone rows stay identity in F and zero in G.
        F = np.eye(size)
        F[IDX_X, IDX_VX] = dt
        F[IDX_Y, IDX_VY] = dt
        F[IDX_X, IDX_YAW] = 0.5 * dt**2 * dax_dyaw
        F[IDX_Y, IDX_YAW] = 0.5 * dt**2 * day_dyaw
        F[IDX_VX, IDX_YAW] = dt * dax_dyaw
        F[IDX_VY, IDX_YAW] = dt * day_dyaw
        F[IDX_X, IDX_BAX] = -0.5 * dt**2 * c
        F[IDX_X, IDX_BAY] = 0.5 * dt**2 * s
        F[IDX_Y, IDX_BAX] = -0.5 * dt**2 * s
        F[IDX_Y, IDX_BAY] = -0.5 * dt**2 * c
        F[IDX_VX, IDX_BAX] = -dt * c
        F[IDX_VX, IDX_BAY] = dt * s
        F[IDX_VY, IDX_BAX] = -dt * s
        F[IDX_VY, IDX_BAY] = -dt * c
        F[IDX_YAW, IDX_BG] = -dt

        G = np.zeros((size, 6), dtype=float)
        G[[IDX_X, IDX_Y], 0:2] = 0.5 * dt**2 * rotation
        G[[IDX_VX, IDX_VY], 0:2] = dt * rotation
        G[IDX_YAW, 2] = dt
        G[IDX_BAX, 3] = np.sqrt(dt)
        G[IDX_BAY, 4] = np.sqrt(dt)
        G[IDX_BG, 5] = np.sqrt(dt)

        # G carries the discrete-time integration factors. The accel/gyro terms
        # below are per-sample IMU measurement noise variances, while the bias
        # terms are random-walk increment variances propagated over one step.
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
        predicted_covariance = 0.5 * (predicted_covariance + predicted_covariance.T) # Ensure symmetry
        return predicted, predicted_covariance

    def update(self, predicted_state: np.ndarray, predicted_covariance: np.ndarray, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply a GNSS position/velocity update."""
        H = self._pad_columns(self.H, predicted_state.shape[0])
        innovation = measurement - H @ predicted_state
        return self._kalman_update(predicted_state, predicted_covariance, innovation, H, self.R)

    @staticmethod
    def _pad_columns(H: np.ndarray, size: int) -> np.ndarray:
        if H.shape[1] == size:
            return H
        padded = np.zeros((H.shape[0], size), dtype=float)
        padded[:, : H.shape[1]] = H
        return padded

    @staticmethod
    def _kalman_update(
        state: np.ndarray, covariance: np.ndarray, innovation: np.ndarray, H: np.ndarray, R: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Joseph-form EKF update for a given innovation and measurement Jacobian."""
        innovation_covariance = H @ covariance @ H.T + R
        # Solve S K^T = (P H^T)^T instead of forming S^{-1} explicitly.
        kalman_gain = np.linalg.solve(
            innovation_covariance.T,
            (covariance @ H.T).T,
        ).T

        updated_state = state + kalman_gain @ innovation
        updated_state[IDX_YAW] = wrap_angle(updated_state[IDX_YAW])
        if state.shape[0] > IDX_CLONE_YAW:
            updated_state[IDX_CLONE_YAW] = wrap_angle(updated_state[IDX_CLONE_YAW])

        identity = np.eye(state.shape[0])
        residual_projector = identity - kalman_gain @ H
        updated_covariance = (
            residual_projector @ covariance @ residual_projector.T
            + kalman_gain @ R @ kalman_gain.T
        )
        updated_covariance = 0.5 * (updated_covariance + updated_covariance.T) # Ensure symmetry
        return updated_state, updated_covariance

    @staticmethod
    def clone_pose(state: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Start a VO frame: append a copy of [x, y, yaw] to the 8-state (replacing any existing clone).

        With x_aug = J x, the clone is fully correlated with the pose: P_aug = J P J^T.
        """
        J = np.zeros((AUGMENTED_STATE_SIZE, STATE_SIZE), dtype=float)
        J[:STATE_SIZE, :STATE_SIZE] = np.eye(STATE_SIZE)
        J[IDX_CLONE_X, IDX_X] = 1.0
        J[IDX_CLONE_Y, IDX_Y] = 1.0
        J[IDX_CLONE_YAW, IDX_YAW] = 1.0
        base_state = state[:STATE_SIZE]
        base_covariance = covariance[:STATE_SIZE, :STATE_SIZE]
        return J @ base_state, J @ base_covariance @ J.T

    @staticmethod
    def drop_clone(state: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Marginalize the clone out (keep the 8-state block)."""
        return state[:STATE_SIZE].copy(), covariance[:STATE_SIZE, :STATE_SIZE].copy()

    @staticmethod
    def vo_measurement_model(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predicted VO delta pose h = [R(yaw_c)^T (p - p_c), wrap(yaw - yaw_c)] and its Jacobian (3 x 11)."""
        d = state[[IDX_X, IDX_Y]] - state[[IDX_CLONE_X, IDX_CLONE_Y]]
        c = np.cos(state[IDX_CLONE_YAW])
        s = np.sin(state[IDX_CLONE_YAW])
        dx_body = c * d[0] + s * d[1]
        dy_body = -s * d[0] + c * d[1]
        predicted = np.array([dx_body, dy_body, wrap_angle(state[IDX_YAW] - state[IDX_CLONE_YAW])])

        H = np.zeros((VO_MEAS_SIZE, state.shape[0]), dtype=float)
        H[0, [IDX_X, IDX_Y]] = [c, s]
        H[1, [IDX_X, IDX_Y]] = [-s, c]
        H[0, [IDX_CLONE_X, IDX_CLONE_Y]] = [-c, -s]
        H[1, [IDX_CLONE_X, IDX_CLONE_Y]] = [s, -c]
        # d(R(yaw_c)^T d)/d(yaw_c) = [dy_body, -dx_body]
        H[0, IDX_CLONE_YAW] = dy_body
        H[1, IDX_CLONE_YAW] = -dx_body
        H[2, IDX_YAW] = 1.0
        H[2, IDX_CLONE_YAW] = -1.0
        return predicted, H

    def vo_innovation(self, state: np.ndarray, covariance: np.ndarray, measurement: np.ndarray, vo_covariance: np.ndarray):
        """VO innovation z - h(x) (yaw wrapped), its covariance H P H^T + R, and H."""
        predicted, H = self.vo_measurement_model(state)
        innovation = measurement - predicted
        innovation[2] = wrap_angle(innovation[2])
        return innovation, H @ covariance @ H.T + vo_covariance, H

    def update_vo(
        self, state: np.ndarray, covariance: np.ndarray, measurement: np.ndarray, vo_covariance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply a VO delta-pose update [dx_body, dy_body, dyaw] against the clone in the augmented state."""
        if state.shape[0] != AUGMENTED_STATE_SIZE:
            raise ValueError("update_vo needs the augmented state; call clone_pose at the VO frame first")
        innovation, _, H = self.vo_innovation(state, covariance, measurement, vo_covariance)
        return self._kalman_update(state, covariance, innovation, H, vo_covariance)


def run_filter(
    imu_measurements: np.ndarray,
    gnss_measurements: np.ndarray,
    gnss_available: np.ndarray,
    dt: float,
    config: EkfConfig | None = None,
    initial_state: np.ndarray | None = None,
    initial_covariance: np.ndarray | None = None,
    vo_from_index: np.ndarray | None = None,
    vo_to_index: np.ndarray | None = None,
    vo_measurements: np.ndarray | None = None,
    vo_covariances: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Run the EKF over a full simulated sequence.

    Besides the posterior `state_estimates`/`covariances`, the result carries the
    prior (after predict, before update) state/covariance at every step, plus the
    GNSS innovation `z - H x_prior` and its covariance `H P_prior H^T + R`
    (NaN on steps without a GNSS update) for consistency checks and animation.

    Visual odometry is fused when `vo_measurements` is given, together with
    `vo_from_index`/`vo_to_index`/`vo_covariances` (as returned by `simulate_scenario`).
    By stochastic cloning: at step `vo_from_index[k]` the pose is cloned into the
    state (after that step's updates), and at `vo_to_index[k]` delta `k` updates
    against the clone (before GNSS), which is then dropped. Only one clone is open
    at a time, so each delta must start at or after the previous one's end. The
    returned histories stay 8-state; `vo_innovations`/`vo_innovation_covariances`
    are per step and NaN where there is no VO update.
    """
    if config is None:
        config = EkfConfig()

    ekf = ImuGnssEkf(config)
    state = np.zeros(STATE_SIZE, dtype=float) if initial_state is None else initial_state.astype(float).copy()
    if initial_covariance is None:
        covariance = np.diag(
            [
                config.initial_position_std**2,
                config.initial_position_std**2,
                config.initial_velocity_std**2,
                config.initial_velocity_std**2,
                config.initial_yaw_std_rad**2,
                config.initial_accel_bias_std**2,
                config.initial_accel_bias_std**2,
                config.initial_gyro_bias_std**2,
            ]
        )
    else:
        covariance = initial_covariance.astype(float).copy()

    num_steps = imu_measurements.shape[0]
    state_history = np.zeros((num_steps, STATE_SIZE), dtype=float)
    covariance_history = np.zeros((num_steps, STATE_SIZE, STATE_SIZE), dtype=float)
    prior_state_history = np.zeros((num_steps, STATE_SIZE), dtype=float)
    prior_covariance_history = np.zeros((num_steps, STATE_SIZE, STATE_SIZE), dtype=float)
    innovation_history = np.full((num_steps, MEAS_SIZE), np.nan, dtype=float)
    innovation_covariance_history = np.full((num_steps, MEAS_SIZE, MEAS_SIZE), np.nan, dtype=float)
    vo_innovation_history = np.full((num_steps, VO_MEAS_SIZE), np.nan, dtype=float)
    vo_innovation_covariance_history = np.full((num_steps, VO_MEAS_SIZE, VO_MEAS_SIZE), np.nan, dtype=float)

    vo_by_to_step: dict[int, int] = {}
    vo_clone_steps: set[int] = set()
    if vo_measurements is not None:
        vo_from_index = np.asarray(vo_from_index, dtype=int)
        vo_to_index = np.asarray(vo_to_index, dtype=int)
        if np.any(vo_to_index <= vo_from_index) or np.any(vo_from_index[1:] < vo_to_index[:-1]):
            raise ValueError("VO deltas must go forward in time and must not overlap (one clone at a time)")
        vo_by_to_step = {int(step): k for k, step in enumerate(vo_to_index)}
        vo_clone_steps = {int(step) for step in vo_from_index}
    clone_step: int | None = None

    for index in range(num_steps):
        if index > 0:
            state, covariance = ekf.predict(state, covariance, imu_measurements[index - 1], dt)
        prior_state_history[index] = state[:STATE_SIZE]
        prior_covariance_history[index] = covariance[:STATE_SIZE, :STATE_SIZE]
        if index in vo_by_to_step:
            k = vo_by_to_step[index]
            if clone_step != vo_from_index[k]:
                raise ValueError(f"VO delta {k} starts at step {vo_from_index[k]}, but no pose was cloned there")
            vo_innovation_history[index], vo_innovation_covariance_history[index], _ = ekf.vo_innovation(
                state, covariance, vo_measurements[k], vo_covariances[k]
            )
            state, covariance = ekf.update_vo(state, covariance, vo_measurements[k], vo_covariances[k])
            state, covariance = ekf.drop_clone(state, covariance)
            clone_step = None
        if gnss_available[index]:
            innovation_history[index] = gnss_measurements[index] - ekf.H @ state[:STATE_SIZE]
            innovation_covariance_history[index] = ekf.H @ covariance[:STATE_SIZE, :STATE_SIZE] @ ekf.H.T + ekf.R
            state, covariance = ekf.update(state, covariance, gnss_measurements[index])
        if index in vo_clone_steps:
            state, covariance = ekf.clone_pose(state, covariance)
            clone_step = index
        state_history[index] = state[:STATE_SIZE]
        covariance_history[index] = covariance[:STATE_SIZE, :STATE_SIZE]

    return {
        "state_estimates": state_history,
        "covariances": covariance_history,
        "prior_state_estimates": prior_state_history,
        "prior_covariances": prior_covariance_history,
        "innovations": innovation_history,
        "innovation_covariances": innovation_covariance_history,
        "vo_innovations": vo_innovation_history,
        "vo_innovation_covariances": vo_innovation_covariance_history,
    }
