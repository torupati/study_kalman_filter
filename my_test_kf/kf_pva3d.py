import numpy as np

class KalmanFilterPVA_RandomAcc3d():
    """
    """

    def __init__(self, _sig_acc:np.array):
        self.sig_acc = _sig_acc
        self._H = np.array([\
        [1, 0, 0, 0, 0, 0, 0, 0, 0], \
        [0, 1, 0, 0, 0, 0, 0, 0, 0], \
        [0, 0, 1, 0, 0, 0, 0, 0, 0]]).reshape(3, 9)

    @property
    def H(self) -> np.array:
        """
        observation matrix
        """
        return self._H

    def F(self, dt: float):
        """
        State transition matrix
        """
        return np.array([\
            [1, 0, 0, dt, 0, 0, 0.5*dt*dt, 0, 0], \
            [0, 1, 0, 0, dt, 0, 0, 0.5*dt*dt, 0], \
            [0, 0, 1, 0, 0, dt, 0, 0, 0.5*dt*dt], \
            [0, 0, 0, 1, 0, 0, dt, 0, 0], \
            [0, 0, 0, 0, 1, 0, 0, dt, 0], \
            [0, 0, 0, 0, 0, 1, 0, 0, dt], \
            [0, 0, 0, 0, 0, 0, 1, 0, 0], \
            [0, 0, 0, 0, 0, 0, 0, 1, 0], \
            [0, 0, 0, 0, 0, 0, 0, 0, 1]])

    def Q(self, dt:float):
        """
        """
        Q = np.zeros([9, 9])
        Q[0:3, 0:3] = 1.0 / 20.0 * np.power(dt, 5) * self.sig_acc # E[p(t)p(t)]
        Q[0:3, 3:6] = Q[3:6, 0:3] = 1.0 / 8.0 * np.power(dt, 4) * self.sig_acc # E[p(t)v(t)]
        Q[0:3, 6:9] = Q[6:9, 0:3] =  np.power(dt, 3) * self.sig_acc * (1.0 / 6.0)# E[p(t)a(t)]
        Q[3:6, 3:6] = 1.0 / 3.0 * np.power(dt, 3) * self.sig_acc # E[v(t)v(t)]
        Q[3:6, 0:3] = Q[3:6, 0:3] = 1.0 / 2.0 * np.power(dt, 2) * self.sig_acc # E[v(t)a(t)]
        Q[6:9, 6:9] = dt * self.sig_acc # E[a(t)a(t)]
        return Q


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from x_generator import generate_true_pos_vel_acc_3d_type1
    from kf_pva_plot import plot_kf_pva3d_states_filter, plot_kf_pva3d_states_smoother
    from kf_pva_plot import plot_kf_pva3d_states_var, plot_kf_pva3d_state_filter

    Fs = 10
    t_end = 10.0

    sig_acc = np.diag([0.4, 0.01, 0.01]) # noise density of acceleration
    R = np.array([\
        [0.6, 0.0, 0.0],
        [0.0, 0.5, -0.0], \
        [0.0, -0.0, 0.2]]) # covariance of position observation

    x_init = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    x_var_init = np.array([\
        [1, 0, 0, 0, 0, 0, 0, 0, 0], \
        [0, 1, 0, 0, 0, 0, 0, 0, 0], \
        [0, 0, 1, 0, 0, 0, 0, 0, 0], \
        [0, 0, 0, 1, 0, 0, 0, 0, 0], \
        [0, 0, 0, 0, 1, 0, 0, 0, 0], \
        [0, 0, 0, 0, 0, 1, 0, 0, 0], \
        [0, 0, 0, 0, 0, 0, 1, 0, 0], \
        [0, 0, 0, 0, 0, 0, 0, 1, 0], \
        [0, 0, 0, 0, 0, 0, 0, 0, 1]])

    # Generate true and observed position
    t_idx = np.arange(0.0, t_end, 1.0/Fs)
    x_true = generate_true_pos_vel_acc_3d_type1(t_idx)
    pos_obs = np.array([[_x[0], 0, 0] + np.random.multivariate_normal(np.zeros(3), R) for _x in x_true])

    kf_pva3d = KalmanFilterPVA_RandomAcc3d(sig_acc)
    # ---
    # Kalman Filter
    x = x_init.reshape(9, 1)
    P = x_var_init
    t_est = []
    x_est = []
    P_est = []
    t_pred, x_pred, P_pred = [], [], []
    t_prev = 0.0
    for i, (t, y) in enumerate(zip(t_idx, pos_obs)):
        # measurement update
        H = kf_pva3d.H
        S = np.dot(np.dot(H, P), H.T) + R  # inovation (pre-fit residual covariance)
        #K = np.linalg.solve(S, np.dot(P, H.T))
        K = np.dot(np.dot(P, H.T), np.linalg.inv(S))
        #print('S=', S, 'K=', K)
        #print('y - H*x=', y.reshape(3,1) - np.dot(H, x))
        x = x + np.dot(K, y.reshape(3, 1) - np.dot(H, x))
        P = np.dot((np.eye(9) - np.dot(K, H)), P)

        # Save data
        t_est.append(t)
        x_est.append(x)
        P_est.append(P)

        if i >= len(t_idx) - 1:
            break # no more time update

        # time update
        dt = t_idx[i+1] - t
        F = kf_pva3d.F(dt)
        Q = kf_pva3d.Q(dt)
        x = np.dot(F, x)
        P = np.dot(np.dot(F, P), F.T) + Q

        # Save data
        t_pred.append(t)
        x_pred.append(x)
        P_pred.append(P)

    # ------
    fig = plot_kf_pva3d_state_filter(t_idx, x_true, t_est, x_est, t_pred, x_pred, pos_obs)
    ofile = 'kf_3d_state_filter.png'
    plt.savefig(ofile)
    print(ofile)

    # ---
    t_smooth = []
    x_smooth = []
    P_smooth = []
    n = len(t_idx)
    print(f'n={n} len(x_est)={len(x_est)}')
    x_smooth.append(x_est[n-1])
    P_smooth.append(P_est[n-1])
    t_smooth.append(t_idx[n-1])
    _x_back = x_est[n-1]
    _P_back = P_est[n-1]
    for _i in range(n-2, -1, -1):
        dt = t_idx[_i+1] - t_idx[_i]
        F = np.array([\
            [1, 0, 0, dt, 0, 0, 0.5*dt*dt, 0, 0], \
            [0, 1, 0, 0, dt, 0, 0, 0.5*dt*dt, 0], \
            [0, 0, 1, 0, 0, dt, 0, 0, 0.5*dt*dt], \
            [0, 0, 0, 1, 0, 0, dt, 0, 0], \
            [0, 0, 0, 0, 1, 0, 0, dt, 0], \
            [0, 0, 0, 0, 0, 1, 0, 0, dt], \
            [0, 0, 0, 0, 0, 0, 1, 0, 0], \
            [0, 0, 0, 0, 0, 0, 0, 1, 0], \
            [0, 0, 0, 0, 0, 0, 0, 0, 1]])
        Q = np.zeros([9, 9])
        Q[0:3, 0:3] = 1.0 / 20.0 * np.power(dt, 5) * sig_acc # E[p(t)p(t)]
        Q[0:3, 3:6] = Q[3:6, 0:3] = 1.0 / 8.0 * np.power(dt, 4) * sig_acc # E[p(t)v(t)]
        Q[0:3, 6:9] = Q[6:9, 0:3] =  np.power(dt, 3) * sig_acc * (1.0 / 6.0)# E[p(t)a(t)]
        Q[3:6, 3:6] = 1.0 / 3.0 * np.power(dt, 3) * sig_acc # E[v(t)v(t)]
        Q[3:6, 0:3] = Q[3:6, 0:3] = 1.0 / 2.0 * np.power(dt, 2) * sig_acc # E[v(t)a(t)]
        Q[6:9, 6:9] = dt * sig_acc # E[a(t)a(t)]

        _x_pred = np.dot(F, x_est[_i])
        _P_pred = np.dot(np.dot(F, P_est[_i]), F.T) + Q
        #print('_i=', _i, ' P_{t|t}=', P_est[_i], ' P_{t+1|t}=', P_est[_i+1])
        J = np.dot(np.dot(P_est[_i], F.T), np.linalg.inv(_P_pred))
        #print('J=', J)
        #print('x_{t+1|T} - x_{t+1|t}=', _x_back - np.dot(F, x_est[_i]))
        _x_back = x_est[_i] + np.dot(J, (_x_back - _x_pred))
        _P_back = P_est[_i] + np.dot(J, np.dot(_P_back - _P_pred, J.T))
        t_smooth.append(t_idx[_i])
        x_smooth.append(_x_back)
        P_smooth.append(_P_back)
        #print('x_back=', _x_back, ' x_est=', x_est[_i], 'pos_true=', pos_true[_i])
        #print('P_back=', _P_back)

    # ------
    # plot state.
    fig = plot_kf_pva3d_states_smoother(t_est, x_est, P_est, x_smooth, P_smooth, t_idx, x_true, pos_obs)
    ofile = 'kf_3d_state_smoother.png'
    fig.savefig(ofile)
    print(ofile)

    # ------
    # Plot variance
    fig = plot_kf_pva3d_states_var(t_est, P_est, t_smooth, P_smooth)
    ofile = 'kf_3d_state_var.png'
    fig.savefig(ofile)
    print(ofile)
