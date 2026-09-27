#!/bin/bash
# Simulate the stationary / forward_back / circle scenarios, run the EKF, and render one movie each.
# Outputs: outputs/scenarios/<scenario>/{nav_log.npz, nav.mp4, ekf_summary.png, ...}
set -eu

FPS="${FPS:-15}"

render() {
  local scenario="$1" total_time="$2" speed="$3"
  local out="./outputs/scenarios/${scenario}"
  uv run python -m imu_gnss_ekf_training.demo \
    --scenario "${scenario}" --total-time "${total_time}" \
    --output-dir "${out}" --save-log "${out}/nav_log.npz"
  uv run python -m imu_gnss_ekf_training.animate "${out}/nav_log.npz" \
    --output "${out}/nav.mp4" --fps "${FPS}" --speed "${speed}"
}

# The three renders are independent, so run them in parallel.
pids=()
render stationary 30 1 & pids+=($!)
render forward_back 30 1 & pids+=($!)
render circle 40 2 & pids+=($!)
for pid in "${pids[@]}"; do wait "${pid}"; done  # fail if any render failed
