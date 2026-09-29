# Sensor Simulation for 2D Navigation

Here we assume a car-like robot with an IMU and GNSS. We simulate sensor observations from the ground-truth position, velocity, attitude (yaw only), and sensor biases.

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
