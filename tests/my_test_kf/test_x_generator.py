import numpy as np

from my_test_kf.x_generator import (
    generate_true_pos_vel_acc_3d_type1,
    generte_true_pos_vel_1d_type1,
    kf_pv_1d_type1_condition,
)


def test_kf_pv_1d_type1_condition_returns_expected_values():
    fs, t_end, sig_acc, sig_pos, x0, x0_var = kf_pv_1d_type1_condition()

    assert fs == 2
    assert t_end == 60.0
    assert sig_acc == 1.0
    assert sig_pos == 0.5
    np.testing.assert_array_equal(x0, [0.0, 0.0])
    np.testing.assert_array_equal(x0_var, [[4.0, 0.0], [0.0, 4.0]])


def test_generte_true_pos_vel_1d_type1_shape_and_start_point():
    t_idx = np.linspace(0.0, 10.0, 21)
    motion_period_time = 4.0

    x_true = generte_true_pos_vel_1d_type1(t_idx, motion_period_time)

    assert x_true.shape == (len(t_idx), 2)
    # default pos_offset=1.0 zeroes the position at t=0
    np.testing.assert_allclose(x_true[0], [0.0, 0.0], atol=1e-12)


def test_generte_true_pos_vel_1d_type1_matches_formula():
    t_idx = np.array([0.0, 1.0, 2.5])
    motion_period_time = 5.0
    pos_offset = 0.3
    w = 2.0 * np.pi / motion_period_time

    x_true = generte_true_pos_vel_1d_type1(t_idx, motion_period_time, pos_offset=pos_offset)

    expected_pos = 2.3 * (np.cos(w * t_idx) - pos_offset)
    expected_vel = 2.3 * w * (-np.sin(w * t_idx))
    np.testing.assert_allclose(x_true[:, 0], expected_pos)
    np.testing.assert_allclose(x_true[:, 1], expected_vel)


def test_generate_true_pos_vel_acc_3d_type1_shape_and_start_point():
    t_idx = np.linspace(0.0, 10.0, 11)

    x_true = generate_true_pos_vel_acc_3d_type1(t_idx)

    assert x_true.shape == (len(t_idx), 9)
    # position starts at 0, velocity starts at 0, acceleration starts at 0
    np.testing.assert_allclose(x_true[0], np.zeros(9), atol=1e-12)
    # only the x-axis (index 0/3/6) is populated; y/z stay zero
    np.testing.assert_array_equal(x_true[:, [1, 2, 4, 5, 7, 8]], 0.0)
