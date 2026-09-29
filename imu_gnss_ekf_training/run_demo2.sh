#!/bin/bash
set -eu

mkdir -p ./outputs/demo2

uv run python -m imu_gnss_ekf_training.record_sensors \
  demo2 \
  --output-dir ./outputs/demo2/ \
  --dt 0.01 \
  --gnss-period 0.5 \
  --save-true-state --total-time 60 \
  --no-show

cat > ./outputs/demo2/initial_state.json <<'JSON'
{
  "x": -5.0,
  "y": 5.0,
  "vx": 0.0,
  "vy": 0.0,
  "yaw_deg": 60.0,
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
  --imu-csv ./outputs/demo2/imu_observations.csv \
  --gnss-csv ./outputs/demo2/gnss_observations.csv \
  --true-state-csv ./outputs/demo2/true_state.csv \
  --output-dir ./outputs/demo2/ \
  --gnss-position-std 3.0 \
  --gnss-velocity-std 3.0 \
  --initial-state-json ./outputs/demo2/initial_state.json
    