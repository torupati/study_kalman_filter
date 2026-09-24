"""Compare the EKF's discrete process noise Q with the exact (Van Loan) Q.

Supports doc/process_noise.md. The Van Loan Q integrates the linearized continuous-time
error dynamics over one step, with IMU white noise of PSD q = sigma^2 * dt (the continuous
equivalent of the per-sample noise used by ImuGnssEkf.predict and simulator.py).

Usage: uv run python -m imu_gnss_ekf_training.misc.compare_process_noise
"""
from __future__ import annotations

import argparse
import logging

import numpy as np

from imu_gnss_ekf_training.ekf import IDX_BAX, IDX_BAY, IDX_BG, IDX_VX, IDX_VY, IDX_X, IDX_Y, IDX_YAW, STATE_SIZE, EkfConfig, ImuGnssEkf

NAMES = ["x", "y", "vx", "vy", "yaw", "bax", "bay", "bg"]

logger = logging.getLogger(__name__)


def expm(matrix: np.ndarray, terms: int = 30) -> np.ndarray:
    """Matrix exponential by scaling and squaring of a Taylor series (NumPy only)."""
    squarings = max(0, int(np.ceil(np.log2(max(np.linalg.norm(matrix, 1), 1e-16)))) + 1)
    scaled = matrix / 2**squarings
    result = np.eye(matrix.shape[0])
    term = np.eye(matrix.shape[0])
    for k in range(1, terms):
        term = term @ scaled / k
        result = result + term
    for _ in range(squarings):
        result = result @ result
    return result


def van_loan_q(state: np.ndarray, imu_sample: np.ndarray, dt: float, config: EkfConfig) -> np.ndarray:
    """Exact discrete Q of the linearized continuous model d(dx)/dt = A dx + L w over one step."""
    yaw = state[IDX_YAW]
    c, s = np.cos(yaw), np.sin(yaw)
    ax_c = imu_sample[0] - state[IDX_BAX]
    ay_c = imu_sample[1] - state[IDX_BAY]

    A = np.zeros((STATE_SIZE, STATE_SIZE))
    A[IDX_X, IDX_VX] = 1.0
    A[IDX_Y, IDX_VY] = 1.0
    A[IDX_VX, IDX_YAW] = -s * ax_c - c * ay_c
    A[IDX_VY, IDX_YAW] = c * ax_c - s * ay_c
    A[IDX_VX, IDX_BAX], A[IDX_VX, IDX_BAY] = -c, s
    A[IDX_VY, IDX_BAX], A[IDX_VY, IDX_BAY] = -s, -c
    A[IDX_YAW, IDX_BG] = -1.0

    L = np.zeros((STATE_SIZE, 6))
    L[[IDX_VX, IDX_VY], 0:2] = np.array([[c, -s], [s, c]])
    L[IDX_YAW, 2] = 1.0
    L[IDX_BAX, 3] = L[IDX_BAY, 4] = L[IDX_BG, 5] = 1.0

    psd = np.diag(
        [
            config.accel_noise_std**2 * dt,
            config.accel_noise_std**2 * dt,
            config.gyro_noise_std**2 * dt,
            config.accel_bias_walk_std**2,
            config.accel_bias_walk_std**2,
            config.gyro_bias_walk_std**2,
        ]
    )

    n = STATE_SIZE
    block = np.zeros((2 * n, 2 * n))
    block[:n, :n] = -A
    block[:n, n:] = L @ psd @ L.T
    block[n:, n:] = A.T
    exp_block = expm(block * dt)
    phi = exp_block[n:, n:].T
    return phi @ exp_block[:n, n:]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare the EKF's G Sigma G^T process noise with the exact Van Loan Q.")
    parser.add_argument("--dt", type=float, default=0.1, help="filter step [s]")
    parser.add_argument("--yaw-deg", type=float, default=np.rad2deg(0.6), help="linearization yaw [deg]")
    parser.add_argument("--accel", type=float, nargs=2, default=[0.3, 0.1], metavar=("AX", "AY"), help="body-frame accel [m/s^2]")
    parser.add_argument("--gyro-z-dps", type=float, default=np.rad2deg(0.2), help="yaw rate [deg/s]")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = EkfConfig()
    dt = args.dt
    state = np.zeros(STATE_SIZE)
    state[IDX_YAW] = np.deg2rad(args.yaw_deg)
    imu_sample = np.array([args.accel[0], args.accel[1], np.deg2rad(args.gyro_z_dps)])

    _, q_code = ImuGnssEkf(config).predict(state, np.zeros((STATE_SIZE, STATE_SIZE)), imu_sample, dt)
    q_exact = van_loan_q(state, imu_sample, dt, config)

    logger.info("dt = %g s, yaw = %.1f deg, accel = %s m/s^2, gyro_z = %.2f deg/s, default EkfConfig",
                dt, args.yaw_deg, args.accel, args.gyro_z_dps)
    logger.info("%-10s%12s%14s", "entry", "code Q", "Van Loan Q")
    for i, j in [(IDX_X, IDX_X), (IDX_X, IDX_VX), (IDX_VX, IDX_VX), (IDX_YAW, IDX_YAW), (IDX_BAX, IDX_BAX), (IDX_BG, IDX_BG),
                 (IDX_VX, IDX_BAX), (IDX_YAW, IDX_BG), (IDX_VX, IDX_YAW), (IDX_X, IDX_YAW)]:
        logger.info("%-10s%12.3e%14.3e", f"{NAMES[i]},{NAMES[j]}", q_code[i, j], q_exact[i, j])


if __name__ == "__main__":
    main()
