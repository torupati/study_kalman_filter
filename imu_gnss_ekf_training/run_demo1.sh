#!/bin/bash
set -eu

mkdir -p ./outputs/demo1

uv run python -m imu_gnss_ekf_training.record_sensors \
  --output-dir ./outputs/demo1/ \
  --dt 0.01 \
  --gnss-period 2.0 \
  --save-true-state --total-time 60 \
  --no-show

cat > ./outputs/demo1/initial_state.json <<'JSON'
{
  "x": -50.0,
  "y": 50.0,
  "vx": 0.0,
  "vy": 0.0,
  "yaw_deg": 90.0,
  "cov_pos_xx": 100.0,
  "cov_pos_yy": 100.0,
  "cov_pos_xy": 0.0,
  "cov_vel_xx": 10.0,
  "cov_vel_yy": 10.0,
  "cov_vel_xy": 0.0,
  "yaw_std_deg": 360.0
}
JSON

uv run python -m imu_gnss_ekf_training.run_ekf_from_csv \
  --imu-csv ./outputs/demo1/imu_observations.csv \
  --gnss-csv ./outputs/demo1/gnss_observations.csv \
  --true-state-csv ./outputs/demo1/true_state.csv \
  --output-dir ./outputs/demo1/ \
  --gnss-position-std 3.0 \
  --gnss-velocity-std 3.0 \
  --initial-state-json ./outputs/demo1/initial_state.json
    