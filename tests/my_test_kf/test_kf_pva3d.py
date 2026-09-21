import numpy as np

from my_test_kf.kf_pva3d import KalmanFilterPVA_RandomAcc3d


def test_H_selects_position_block():
    kf = KalmanFilterPVA_RandomAcc3d(np.diag([0.4, 0.01, 0.01]))

    assert kf.H.shape == (3, 9)
    np.testing.assert_array_equal(kf.H[:, 0:3], np.eye(3))
    np.testing.assert_array_equal(kf.H[:, 3:9], np.zeros((3, 6)))


def test_F_matches_constant_acceleration_transition():
    kf = KalmanFilterPVA_RandomAcc3d(np.diag([0.4, 0.01, 0.01]))
    dt = 0.25

    F = kf.F(dt)

    expected = np.eye(9)
    expected[0:3, 3:6] = dt * np.eye(3)
    expected[0:3, 6:9] = 0.5 * dt * dt * np.eye(3)
    expected[3:6, 6:9] = dt * np.eye(3)
    np.testing.assert_allclose(F, expected)


def test_Q_scales_with_noise_density_and_dt():
    sig_acc = np.diag([0.4, 0.01, 0.01])
    kf = KalmanFilterPVA_RandomAcc3d(sig_acc)
    dt = 0.5

    Q = kf.Q(dt)

    assert Q.shape == (9, 9)
    np.testing.assert_allclose(Q[0:3, 0:3], (1.0 / 20.0) * dt**5 * sig_acc)
    np.testing.assert_allclose(Q[3:6, 3:6], (1.0 / 3.0) * dt**3 * sig_acc)
    np.testing.assert_allclose(Q[6:9, 6:9], dt * sig_acc)
    np.testing.assert_allclose(Q[0:3, 6:9], (1.0 / 6.0) * dt**3 * sig_acc)
    np.testing.assert_allclose(Q[6:9, 0:3], (1.0 / 6.0) * dt**3 * sig_acc)
    np.testing.assert_allclose(Q[0:3, 3:6], (1.0 / 8.0) * dt**4 * sig_acc)
    np.testing.assert_allclose(Q[3:6, 0:3], (1.0 / 8.0) * dt**4 * sig_acc)
    np.testing.assert_allclose(Q[3:6, 6:9], 0.5 * dt**2 * sig_acc)
    np.testing.assert_allclose(Q[6:9, 3:6], 0.5 * dt**2 * sig_acc)
    np.testing.assert_allclose(Q, Q.T)
