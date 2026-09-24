# Sensor Simulation for 2D Navigation

Here we suppose to have car robot which have inertial sensor and GNSS on it. We simulate its sensor with ground true position, velocity, attitude(here only yaw is used) and sensor biases.

```bash
uv run python -m imu_gnss_ekf_training.record_sensors demo1
```

### Input

You can set sceinario setup

- time length
- start datetime


### Output


###


```bash
IMU CSV  : outputs/imu_gnss_ekf_training/imu_observations.csv  (6001 rows)
GNSS CSV : outputs/imu_gnss_ekf_training/gnss_observations.csv  (100 rows)
IMU plot : outputs/imu_gnss_ekf_training/imu_timeseries.png
Pos plot : outputs/imu_gnss_ekf_training/gnss_position_2d.png
```
