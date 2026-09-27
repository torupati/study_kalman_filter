from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ekf import IDX_BAX, IDX_BAY, IDX_BG, IDX_VX, IDX_VY, IDX_X, IDX_Y, IDX_YAW, STATE_SIZE


@dataclass(frozen=True)
class SimulatorConfig:
    """Simulator parameters.

    The bias walk values are random-walk noise densities, so the per-step
    increment standard deviation is `walk_std * sqrt(dt)`.
    """

    total_time: float = 60.0
    dt: float = 0.1
    gnss_period: float = 0.5
    accel_noise_std: float = 0.12
    gyro_noise_std: float = 0.015
    gnss_position_std: float = 1.5
    gnss_velocity_std: float = 0.35
    accel_bias_walk_std: float = 0.01
    gyro_bias_walk_std: float = 0.0025
    gnss_dropout_probability: float = 0.15
    initial_accel_bias: tuple[float, float] = (0.08, -0.05)
    initial_gyro_bias: float = 0.015
    seed: int = 7
    scenario: str = "demo1"


def rotation_matrix(yaw: float) -> np.ndarray:
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=float)


def truth_inputs_demo1(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Generate smooth body-frame acceleration and yaw-rate commands."""
    accel_x = 0.45 * np.sin(0.18 * times) + 0.08 * np.cos(0.05 * times)
    accel_y = 0.15 * np.cos(0.11 * times) + 0.03 * np.sin(0.41 * times)
    yaw_rate = 0.16 + 0.08 * np.sin(0.07 * times) + 0.02 * np.cos(0.23 * times)
    return np.column_stack((accel_x, accel_y)), yaw_rate


def truth_inputs_demo2(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Body-frame acceleration/yaw-rate commands for a there-and-back maneuver:

    accelerate, cruise straight for 30 m, half-circle turn (coordinated turn), cruise
    straight back, decelerate to a stop.
    """
    cruise_speed = 3.0
    accel_mag = 0.5
    turn_radius = 8.0
    straight_distance = 30.0

    accel_time = cruise_speed / accel_mag
    accel_distance = 0.5 * accel_mag * accel_time**2
    cruise_time = (straight_distance - accel_distance) / cruise_speed
    omega = cruise_speed / turn_radius
    turn_time = np.pi / omega

    t_accel_end = accel_time
    t_cruise_end = t_accel_end + cruise_time
    t_turn_end = t_cruise_end + turn_time
    t_cruise_back_end = t_turn_end + cruise_time
    t_decel_end = t_cruise_back_end + accel_time

    accel_x = np.zeros_like(times)
    accel_y = np.zeros_like(times)
    yaw_rate = np.zeros_like(times)

    accelerating = times < t_accel_end
    accel_x[accelerating] = accel_mag

    turning = (times >= t_cruise_end) & (times < t_turn_end)
    accel_y[turning] = cruise_speed * omega
    yaw_rate[turning] = omega

    decelerating = (times >= t_cruise_back_end) & (times < t_decel_end)
    accel_x[decelerating] = -accel_mag

    return np.column_stack((accel_x, accel_y)), yaw_rate


def truth_inputs_stationary(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vehicle stays at the origin: zero body-frame acceleration and yaw rate."""
    return np.zeros((times.shape[0], 2)), np.zeros_like(times)


def truth_inputs_forward_back(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stand still, drive 5 m forward (+x), stop, reverse 5 m back to the origin, stand still.

    Each 5 m leg is a bang-bang profile (accelerate, then decelerate to rest) with no
    turning, so the vehicle reverses rather than turning around and yaw stays at 0.
    """
    leg_distance = 5.0
    hold_time = 3.0
    # Phase boundaries are kept on round times so they fall exactly on the sample grid;
    # otherwise the discretized accel/decel halves don't cancel and the vehicle creeps.
    half_leg_time = 2.5
    accel_mag = leg_distance / half_leg_time**2  # accel+decel halves: a * t^2 = distance

    t_forward = hold_time
    t_backward = t_forward + 2.0 * half_leg_time + hold_time

    t = times + 1e-9  # guard against k * dt landing just below a boundary
    accel_x = np.zeros_like(times)
    for start, sign in ((t_forward, 1.0), (t_backward, -1.0)):
        accel_x[(t >= start) & (t < start + half_leg_time)] = sign * accel_mag
        accel_x[(t >= start + half_leg_time) & (t < start + 2.0 * half_leg_time)] = -sign * accel_mag
    return np.column_stack((accel_x, np.zeros_like(times))), np.zeros_like(times)


def truth_inputs_circle(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drive counter-clockwise on a 10 m radius circle centered at (0, 10), starting from rest.

    Speed ramps up to 2 m/s and then stays constant. Staying on one circle with speed
    v(t) needs body-frame acceleration (dv/dt, v^2 / R) and yaw rate v / R.
    """
    radius = 10.0
    cruise_speed = 2.0
    accel_mag = 0.5
    speed = np.minimum(accel_mag * times, cruise_speed)
    accel_x = np.where(accel_mag * times < cruise_speed, accel_mag, 0.0)
    return np.column_stack((accel_x, speed**2 / radius)), speed / radius


_TRUTH_INPUTS_BY_SCENARIO = {
    "demo1": truth_inputs_demo1,
    "demo2": truth_inputs_demo2,
    "stationary": truth_inputs_stationary,
    "forward_back": truth_inputs_forward_back,
    "circle": truth_inputs_circle,
}
SCENARIOS = tuple(_TRUTH_INPUTS_BY_SCENARIO)


def truth_inputs(times: np.ndarray, scenario: str = "demo1") -> tuple[np.ndarray, np.ndarray]:
    """Generate body-frame acceleration/yaw-rate commands for the given scenario."""
    try:
        generator = _TRUTH_INPUTS_BY_SCENARIO[scenario]
    except KeyError:
        raise ValueError(f"Unknown scenario {scenario!r}; choose from {sorted(_TRUTH_INPUTS_BY_SCENARIO)}.") from None
    return generator(times)


def simulate_scenario(config: SimulatorConfig | None = None) -> dict[str, np.ndarray | float]:
    """Simulate truth, IMU measurements, and GNSS measurements."""
    if config is None:
        config = SimulatorConfig()

    rng = np.random.default_rng(config.seed)
    num_steps = int(np.floor(config.total_time / config.dt)) + 1
    times = np.arange(num_steps, dtype=float) * config.dt
    true_accel_body, true_yaw_rate = truth_inputs(times, config.scenario)

    states = np.zeros((num_steps, STATE_SIZE), dtype=float)
    accel_biases = np.zeros((num_steps, 2), dtype=float)
    gyro_biases = np.zeros(num_steps, dtype=float)
    accel_biases[0] = np.array(config.initial_accel_bias, dtype=float)
    gyro_biases[0] = config.initial_gyro_bias

    for index in range(num_steps - 1):
        yaw = states[index, IDX_YAW]
        accel_nav = rotation_matrix(yaw) @ true_accel_body[index]

        states[index + 1, IDX_X] = states[index, IDX_X] + states[index, IDX_VX] * config.dt + 0.5 * accel_nav[0] * config.dt**2
        states[index + 1, IDX_Y] = states[index, IDX_Y] + states[index, IDX_VY] * config.dt + 0.5 * accel_nav[1] * config.dt**2
        states[index + 1, IDX_VX] = states[index, IDX_VX] + accel_nav[0] * config.dt
        states[index + 1, IDX_VY] = states[index, IDX_VY] + accel_nav[1] * config.dt
        states[index + 1, IDX_YAW] = np.arctan2(np.sin(yaw + true_yaw_rate[index] * config.dt),
                                                np.cos(yaw + true_yaw_rate[index] * config.dt))

        accel_biases[index + 1] = accel_biases[index] + config.accel_bias_walk_std * np.sqrt(config.dt) * rng.standard_normal(2)
        gyro_biases[index + 1] = gyro_biases[index] + config.gyro_bias_walk_std * np.sqrt(config.dt) * rng.standard_normal()

    states[:, IDX_BAX] = accel_biases[:, 0]
    states[:, IDX_BAY] = accel_biases[:, 1]
    states[:, IDX_BG] = gyro_biases

    imu_measurements = np.column_stack(
        (
            true_accel_body[:, 0] + accel_biases[:, 0] + config.accel_noise_std * rng.standard_normal(num_steps),
            true_accel_body[:, 1] + accel_biases[:, 1] + config.accel_noise_std * rng.standard_normal(num_steps),
            true_yaw_rate + gyro_biases + config.gyro_noise_std * rng.standard_normal(num_steps),
        )
    )

    gnss_stride = max(1, int(round(config.gnss_period / config.dt)))
    gnss_available = np.zeros(num_steps, dtype=bool)
    gnss_available[::gnss_stride] = True
    gnss_available &= rng.random(num_steps) >= config.gnss_dropout_probability
    gnss_available[0] = True

    gnss_measurements = np.column_stack(
        (
            states[:, IDX_X] + config.gnss_position_std * rng.standard_normal(num_steps),
            states[:, IDX_Y] + config.gnss_position_std * rng.standard_normal(num_steps),
            states[:, IDX_VX] + config.gnss_velocity_std * rng.standard_normal(num_steps),
            states[:, IDX_VY] + config.gnss_velocity_std * rng.standard_normal(num_steps),
        )
    )
    gnss_measurements[~gnss_available] = np.nan

    return {
        "time": times,
        "dt": config.dt,
        "truth_states": states,
        "true_accel_body": true_accel_body,
        "true_yaw_rate": true_yaw_rate,
        "imu_measurements": imu_measurements,
        "gnss_measurements": gnss_measurements,
        "gnss_available": gnss_available,
    }
