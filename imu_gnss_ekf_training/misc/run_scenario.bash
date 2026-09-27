#!/bin/bash


OUTPUT_DIR=./outputs/scenarios2
IMU_INTERVAL=0.01
GNSS_INTERVAL=1.0

mkdir -p $OUTPUT_DIR/stationary
mkdir -p $OUTPUT_DIR/forward_back
mkdir -p $OUTPUT_DIR/circle

#Stationary (30 s, real time)
uv run python -m imu_gnss_ekf_training.demo \
  --scenario stationary --total-time 30 --dt $IMU_INTERVAL \
  --gnss-period $GNSS_INTERVAL \
  --output-dir $OUTPUT_DIR/stationary --save-log $OUTPUT_DIR/stationary/nav_log.npz
uv run python -m imu_gnss_ekf_training.animate $OUTPUT_DIR/stationary/nav_log.npz \
  --output $OUTPUT_DIR/stationary/nav.mp4 --fps 15 --speed 1

#Forward 5 m / back 5 m (30 s, real time)
uv run python -m imu_gnss_ekf_training.demo \
  --scenario forward_back --total-time 30 --dt $IMU_INTERVAL \
  --gnss-period $GNSS_INTERVAL \
  --output-dir $OUTPUT_DIR/forward_back --save-log $OUTPUT_DIR/forward_back/nav_log.npz
uv run python -m imu_gnss_ekf_training.animate $OUTPUT_DIR/forward_back/nav_log.npz \
  --output $OUTPUT_DIR/forward_back/nav.mp4 --fps 15 --speed 1

#Circle (40 s, 2× speed, so a 20 s movie)
uv run python -m imu_gnss_ekf_training.demo \
  --scenario circle --total-time 40 --dt $IMU_INTERVAL \
  --gnss-period $GNSS_INTERVAL \
  --output-dir $OUTPUT_DIR/circle --save-log $OUTPUT_DIR/circle/nav_log.npz
uv run python -m imu_gnss_ekf_training.animate $OUTPUT_DIR/circle/nav_log.npz \
  --output $OUTPUT_DIR/circle/nav.mp4 --fps 15 --speed 2
