import unittest

import numpy as np

from imu_gnss_ekf_training import EkfConfig, SimulatorConfig, run_filter, simulate_scenario
from imu_gnss_ekf_training.ekf import IDX_BG, IDX_BAX, IDX_BAY, IDX_VX, IDX_VY, IDX_X, IDX_Y


class ImuGnssEkfTrainingTests(unittest.TestCase):
    def test_simulator_generates_dropout_and_expected_shapes(self):
        scenario = simulate_scenario(
            SimulatorConfig(total_time=12.0, dt=0.1, gnss_period=0.5, gnss_dropout_probability=0.4, seed=11)
        )

        num_steps = len(scenario["time"])
        self.assertEqual(scenario["truth_states"].shape, (num_steps, 8))
        self.assertEqual(scenario["imu_measurements"].shape, (num_steps, 3))
        self.assertEqual(scenario["gnss_measurements"].shape, (num_steps, 4))
        self.assertEqual(scenario["gnss_available"].shape, (num_steps,))
        self.assertTrue(np.isnan(scenario["gnss_measurements"][~scenario["gnss_available"]]).all())
        self.assertGreater(np.count_nonzero(~scenario["gnss_available"]), 0)

    def test_ekf_runs_and_tracks_state_reasonably(self):
        simulator_config = SimulatorConfig(total_time=40.0, dt=0.1, gnss_period=0.5, gnss_dropout_probability=0.15, seed=7)
        scenario = simulate_scenario(simulator_config)
        result = run_filter(
            imu_measurements=scenario["imu_measurements"],
            gnss_measurements=scenario["gnss_measurements"],
            gnss_available=scenario["gnss_available"],
            dt=scenario["dt"],
            config=EkfConfig(
                accel_noise_std=simulator_config.accel_noise_std,
                gyro_noise_std=simulator_config.gyro_noise_std,
                accel_bias_walk_std=simulator_config.accel_bias_walk_std,
                gyro_bias_walk_std=simulator_config.gyro_bias_walk_std,
                gnss_position_std=simulator_config.gnss_position_std,
                gnss_velocity_std=simulator_config.gnss_velocity_std,
            ),
        )

        estimates = result["state_estimates"]
        truth = scenario["truth_states"]

        self.assertTrue(np.isfinite(estimates).all())
        self.assertTrue(np.isfinite(result["covariances"]).all())

        final_position_error = np.linalg.norm(estimates[-1, [IDX_X, IDX_Y]] - truth[-1, [IDX_X, IDX_Y]])
        final_velocity_error = np.linalg.norm(estimates[-1, [IDX_VX, IDX_VY]] - truth[-1, [IDX_VX, IDX_VY]])
        final_accel_bias_error = np.linalg.norm(estimates[-1, [IDX_BAX, IDX_BAY]] - truth[-1, [IDX_BAX, IDX_BAY]])
        final_gyro_bias_error = abs(estimates[-1, IDX_BG] - truth[-1, IDX_BG])

        self.assertLess(final_position_error, 5.0)
        self.assertLess(final_velocity_error, 1.0)
        self.assertLess(final_accel_bias_error, 0.35)
        self.assertLess(final_gyro_bias_error, 0.05)


if __name__ == "__main__":
    unittest.main()
