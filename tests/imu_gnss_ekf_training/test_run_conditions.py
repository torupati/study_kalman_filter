import tomllib

import numpy as np
import pytest

from imu_gnss_ekf_training import EkfConfig, SimulatorConfig
from imu_gnss_ekf_training.run_conditions import load_run_conditions, save_run_conditions


def test_run_conditions_round_trip(tmp_path):
    simulator_config = SimulatorConfig(scenario="circle", total_time=42.0, initial_accel_bias=(0.1, -0.2), seed=3)
    ekf_config = EkfConfig(gnss_position_std=2.5, initial_yaw_std_rad=np.deg2rad(30.0))

    path = save_run_conditions(tmp_path / "run_conditions.toml", simulator_config, ekf_config)
    loaded_simulator_config, loaded_ekf_config = load_run_conditions(path)

    assert loaded_simulator_config == simulator_config
    assert loaded_ekf_config == ekf_config


def test_simulator_noise_std_follows_density_and_dt():
    config = SimulatorConfig(dt=0.01, accel_noise_density=0.02, gyro_noise_density=0.003)
    assert config.accel_noise_std == pytest.approx(0.2)
    assert config.gyro_noise_std == pytest.approx(0.03)
    # defaults keep the historical per-sample values at dt = 0.1 s
    assert SimulatorConfig(dt=0.1).accel_noise_std == pytest.approx(0.12)
    assert SimulatorConfig(dt=0.1).gyro_noise_std == pytest.approx(0.015)


def test_run_conditions_contains_density_and_std(tmp_path):
    simulator_config = SimulatorConfig(dt=0.01, accel_noise_density=0.02)
    path = save_run_conditions(tmp_path / "run_conditions.toml", simulator_config, EkfConfig())
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    assert data["simulator"]["accel_noise_density"] == pytest.approx(0.02)
    assert data["simulator_derived"]["accel_noise_std"] == pytest.approx(0.2)
    assert "accel_noise_density" in data["ekf_derived"]


def test_load_run_conditions_accepts_legacy_std(tmp_path):
    path = save_run_conditions(tmp_path / "run_conditions.toml", SimulatorConfig(dt=0.01), EkfConfig())
    text = path.read_text(encoding="utf-8").replace("accel_noise_density = ", "accel_noise_std = ", 1)
    text = text.replace("accel_noise_std = ", "accel_noise_std = 0.5  # ", 1)
    path.write_text(text, encoding="utf-8")
    simulator_config, _ = load_run_conditions(path)
    assert simulator_config.accel_noise_std == pytest.approx(0.5)
