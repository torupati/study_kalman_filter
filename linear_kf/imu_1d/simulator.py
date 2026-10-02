"""1D truth trajectory with accelerometer (IMU) and position measurements.

The true acceleration is sampled at the IMU rate and held constant over each IMU
step (zero-order hold), and position/velocity are integrated exactly under that
assumption. The Kalman filter in ``kf.py`` uses the same discretization, so the
only model errors are the ones deliberately added here (noise and bias).
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class SimConfig:
    scenario: str = "sine"  # key of SCENARIOS
    duration: float = 60.0  # [s]
    imu_rate: float = 100.0  # [Hz]
    pos_rate: float = 1.0  # [Hz], must divide imu_rate
    accel_noise_density: float = 0.05  # [m/s^2/sqrt(Hz)]
    accel_bias: float = 0.0  # [m/s^2], constant, not modelled by the 2-state KF
    pos_noise_std: float = 0.5  # [m]
    initial_position: float = 0.0  # [m]
    initial_velocity: float | None = None  # [m/s], None: the scenario's own initial velocity
    pos_outage: tuple[float, float] | None = None  # (start, end) [s] with no position measurements

    @property
    def dt(self) -> float:
        return 1.0 / self.imu_rate

    @property
    def v0(self) -> float:
        """True initial velocity: initial_velocity if set, else the scenario's default."""
        return SCENARIOS[self.scenario].initial_velocity if self.initial_velocity is None else self.initial_velocity

    @property
    def accel_noise_std(self) -> float:
        """Per-sample accelerometer noise std: density / sqrt(dt)."""
        return self.accel_noise_density / np.sqrt(self.dt)

    @property
    def pos_decimation(self) -> int:
        n = self.imu_rate / self.pos_rate
        if abs(n - round(n)) > 1e-9:
            raise ValueError("imu_rate must be an integer multiple of pos_rate")
        return int(round(n))


@dataclass
class SimResult:
    t: np.ndarray  # (N,) IMU sample times
    pos_true: np.ndarray  # (N,)
    vel_true: np.ndarray  # (N,)
    acc_true: np.ndarray  # (N,) applied over [t_k, t_k + dt)
    acc_meas: np.ndarray  # (N,) accelerometer output
    pos_meas: np.ndarray  # (N,) NaN where no position measurement
    config: SimConfig

    @property
    def pos_available(self) -> np.ndarray:
        return ~np.isnan(self.pos_meas)


def accel_sine(t: np.ndarray) -> np.ndarray:
    """Smooth back-and-forth speed changes: two sinusoids (periods 20 s and 7 s). Starts at 1 m/s."""
    return 0.5 * np.sin(2.0 * np.pi * t / 20.0) + 0.3 * np.sin(2.0 * np.pi * t / 7.0)


def accel_stationary(t: np.ndarray) -> np.ndarray:
    """At rest the whole time."""
    return np.zeros_like(t)


# (duration [s], acceleration [m/s^2]) phases of the stop_and_go scenario. Every phase boundary is a
# whole number of seconds so it falls on the IMU sample grid, and each accelerate/decelerate pair has
# the same |a| * duration, so the zero-order-hold velocity returns exactly to 0 at every stop.
STOP_AND_GO_PHASES: tuple[tuple[float, float], ...] = (
    (5.0, 0.0),  # rest
    (4.0, 1.0), (6.0, 0.0), (4.0, -1.0), (3.0, 0.0),  # to 4 m/s, cruise, stop, rest
    (2.0, 1.5), (8.0, 0.0), (2.0, -1.5), (3.0, 0.0),  # to 3 m/s
    (5.0, 0.5), (4.0, 0.0), (5.0, -0.5),  # to 2.5 m/s, then stop for good
)


def accel_stop_and_go(t: np.ndarray) -> np.ndarray:
    """Start at rest, then repeat accelerate / cruise / decelerate / rest, and end at rest."""
    tt = np.round(t, 9)  # k * dt is not exact in floating point; snap so boundaries land on the right sample
    acc = np.zeros_like(t)
    start = 0.0
    for length, a in STOP_AND_GO_PHASES:
        acc[(tt >= start) & (tt < start + length)] = a
        start += length
    return acc


@dataclass(frozen=True)
class Scenario:
    acceleration: Callable[[np.ndarray], np.ndarray]
    initial_velocity: float  # [m/s]
    description: str


SCENARIOS: dict[str, Scenario] = {
    "sine": Scenario(accel_sine, 1.0, "smooth speed changes, always moving"),
    "stationary": Scenario(accel_stationary, 0.0, "at rest the whole time"),
    "stop_and_go": Scenario(accel_stop_and_go, 0.0, "rest, then accelerate / cruise / decelerate / rest three times, end at rest"),
}


def simulate(config: SimConfig, rng: np.random.Generator) -> SimResult:
    dt = config.dt
    n = int(round(config.duration * config.imu_rate)) + 1
    t = np.arange(n) * dt
    acc_true = SCENARIOS[config.scenario].acceleration(t)

    pos_true = np.empty(n)
    vel_true = np.empty(n)
    pos_true[0], vel_true[0] = config.initial_position, config.v0
    for k in range(n - 1):
        pos_true[k + 1] = pos_true[k] + vel_true[k] * dt + 0.5 * acc_true[k] * dt * dt
        vel_true[k + 1] = vel_true[k] + acc_true[k] * dt

    acc_meas = acc_true + config.accel_bias + rng.normal(0.0, config.accel_noise_std, n)

    pos_meas = np.full(n, np.nan)
    idx = np.arange(0, n, config.pos_decimation)
    if config.pos_outage is not None:
        start, end = config.pos_outage
        idx = idx[(t[idx] < start) | (t[idx] >= end)]
    pos_meas[idx] = pos_true[idx] + rng.normal(0.0, config.pos_noise_std, len(idx))

    return SimResult(t, pos_true, vel_true, acc_true, acc_meas, pos_meas, config)
