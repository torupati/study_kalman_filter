"""Save/load the simulator + EKF settings of a run as a human-readable TOML text file.

The file holds every `SimulatorConfig` and `EkfConfig` field, so a run can be
reproduced with `load_run_conditions` (the simulator is deterministic given `seed`).
IMU white noise appears both as a density and as the per-sample std at the run's
`dt` (std = density / sqrt(dt)); the `*_derived` sections are informational only
and are ignored on load.
"""

from __future__ import annotations

import dataclasses
import tomllib
from pathlib import Path

import numpy as np

from .ekf import EkfConfig
from .simulator import SimulatorConfig

SIMULATOR_COMMENTS = {
    "scenario": "truth trajectory (motion pattern)",
    "total_time": "[s]",
    "dt": "[s] IMU sample period",
    "gnss_period": "[s] GNSS sample period",
    "gnss_dropout_probability": "probability that a GNSS epoch is dropped",
    "accel_noise_density": "[m/s^2/sqrt(Hz)] accelerometer white noise density",
    "gyro_noise_density": "[rad/s/sqrt(Hz)] gyro white noise density",
    "accel_bias_walk_std": "[m/s^2/sqrt(s)] accelerometer bias random walk density",
    "gyro_bias_walk_std": "[rad/s/sqrt(s)] gyro bias random walk density",
    "initial_accel_bias": "[m/s^2] true accelerometer bias at t=0 (body x, y)",
    "initial_gyro_bias": "[rad/s] true gyro bias at t=0",
    "gnss_position_std": "[m] GNSS position noise",
    "gnss_velocity_std": "[m/s] GNSS velocity noise",
    "seed": "random seed",
    "forward_back_distance": "[m] one-way travel distance (forward_back scenario only)",
    "initial_position": "[m] true start position in the world frame (x, y)",
    "initial_yaw": "[rad] true start heading in the world frame",
    "vo_period": "[s] visual odometry frame period",
    "vo_translation_std": "[m] VO translation noise per frame (floor)",
    "vo_translation_std_per_m": "[m/m] VO translation noise per meter moved",
    "vo_yaw_std": "[rad] VO yaw noise per frame (floor)",
    "vo_yaw_std_per_rad": "[rad/rad] VO yaw noise per radian turned",
    "vo_dropout_probability": "probability that a VO frame loses tracking",
}

EKF_COMMENTS = {
    "accel_noise_std": "[m/s^2] accelerometer white noise, per sample",
    "gyro_noise_std": "[rad/s] gyro white noise, per sample",
    "accel_bias_walk_std": "[m/s^2/sqrt(s)] accelerometer bias random walk density",
    "gyro_bias_walk_std": "[rad/s/sqrt(s)] gyro bias random walk density",
    "gnss_position_std": "[m] assumed GNSS position noise",
    "gnss_velocity_std": "[m/s] assumed GNSS velocity noise",
    "initial_position_std": "[m]",
    "initial_velocity_std": "[m/s]",
    "initial_yaw_std_rad": "[rad]",
    "initial_accel_bias_std": "[m/s^2]",
    "initial_gyro_bias_std": "[rad/s]",
}


def _format_value(value) -> str:
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | np.integer):
        return str(int(value))
    if isinstance(value, tuple | list):
        return "[" + ", ".join(_format_value(item) for item in value) + "]"
    return repr(float(value))


def _format_section(name: str, config, comments: dict[str, str]) -> list[str]:
    lines = [f"[{name}]"]
    for field in dataclasses.fields(config):
        line = f"{field.name} = {_format_value(getattr(config, field.name))}"
        comment = comments.get(field.name)
        lines.append(f"{line}  # {comment}" if comment else line)
    return lines


def _format_derived_section(name: str, rows: list[tuple[str, float, str]]) -> list[str]:
    return [f"[{name}]", *(f"{key} = {_format_value(value)}  # {comment}" for key, value, comment in rows)]


def format_run_conditions(simulator_config: SimulatorConfig, ekf_config: EkfConfig) -> str:
    lines = [
        "# IMU/GNSS EKF demo run conditions",
        f"# IMU rate: {1.0 / simulator_config.dt:g} Hz, GNSS rate: {1.0 / simulator_config.gnss_period:g} Hz",
        f"# simulated gyro noise: {np.rad2deg(simulator_config.gyro_noise_std):g} deg/s per sample, "
        f"initial gyro bias: {np.rad2deg(simulator_config.initial_gyro_bias):g} deg/s",
        "",
        "# (1) simulation: sensors, motion, duration",
        *_format_section("simulator", simulator_config, SIMULATOR_COMMENTS),
        "",
        "# derived from [simulator] at dt (std = density / sqrt(dt)); not read back",
        *_format_derived_section("simulator_derived", [
            ("imu_rate_hz", 1.0 / simulator_config.dt, "[Hz]"),
            ("gnss_rate_hz", 1.0 / simulator_config.gnss_period, "[Hz]"),
            ("accel_noise_std", simulator_config.accel_noise_std, "[m/s^2] per-sample accelerometer white noise"),
            ("gyro_noise_std", simulator_config.gyro_noise_std, "[rad/s] per-sample gyro white noise"),
        ]),
        "",
        "# (2) estimation: sensor noise model assumed by the EKF",
        *_format_section("ekf", ekf_config, EKF_COMMENTS),
        "",
        "# EKF noise expressed as densities at the simulator dt (density = std * sqrt(dt)); not read back",
        *_format_derived_section("ekf_derived", [
            ("accel_noise_density", ekf_config.accel_noise_std * np.sqrt(simulator_config.dt), "[m/s^2/sqrt(Hz)]"),
            ("gyro_noise_density", ekf_config.gyro_noise_std * np.sqrt(simulator_config.dt), "[rad/s/sqrt(Hz)]"),
        ]),
        "",
    ]
    return "\n".join(lines)


def save_run_conditions(output_path: str | Path, simulator_config: SimulatorConfig, ekf_config: EkfConfig) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_run_conditions(simulator_config, ekf_config), encoding="utf-8")
    return output_path


def load_run_conditions(path: str | Path) -> tuple[SimulatorConfig, EkfConfig]:
    with Path(path).open("rb") as file:
        data = tomllib.load(file)
    simulator_values = dict(data["simulator"])
    # files written before the switch to densities store the per-sample std instead
    dt = simulator_values.get("dt", SimulatorConfig().dt)
    for kind in ("accel", "gyro"):
        std = simulator_values.pop(f"{kind}_noise_std", None)
        if std is not None:
            simulator_values.setdefault(f"{kind}_noise_density", std * np.sqrt(dt))
    simulator_values["initial_accel_bias"] = tuple(simulator_values["initial_accel_bias"])
    if "initial_position" in simulator_values:
        simulator_values["initial_position"] = tuple(simulator_values["initial_position"])
    return SimulatorConfig(**simulator_values), EkfConfig(**data["ekf"])
