import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter

from x_generator import generate_true_pos_vel_acc_3d_type1
from kf_pva_plot import plot_kf_pva3d_states_filter, plot_kf_pva3d_states_smoother

from kf_pva3d import KalmanFilterPVA_RandomAcc3d

np.random.seed(0)

def run_pykalman_kalman_filter(dt, sig_acc, x_init, x_var_init, pos_obs, R):
    # ---
    # Kalman Filter
    kf_pav3d = KalmanFilterPVA_RandomAcc3d(sig_acc)
    H = kf_pav3d.H
    F = kf_pav3d.F(dt)
    Q = kf_pav3d.Q(dt)

    # ------
    kf = KalmanFilter(transition_matrices = F,
                      observation_matrices = H,
                      transition_covariance = Q,
                      observation_covariance = R,
                      initial_state_mean = x_init,
                      initial_state_covariance = x_var_init)
    #kf = KalmanFilter(transition_matrices=F, transition_covariance=Q, n_dim_obs = 3)
    x_est, P_est = kf.filter(pos_obs)
    return x_est, P_est

def run_pykalman_kalman_smoother(dt, sig_acc, x_init, x_var_init, pos_obs, R):
    # ---
    # Kalman Filter
    kf_pav3d = KalmanFilterPVA_RandomAcc3d(sig_acc)
    H = kf_pav3d.H
    F = kf_pav3d.F(dt)
    Q = kf_pav3d.Q(dt)

    # ------
    kf = KalmanFilter(transition_matrices = F,
                      observation_matrices = H,
                      transition_covariance = Q,
                      observation_covariance = R,
                      initial_state_mean = x_init,
                      initial_state_covariance = x_var_init)
    #kf = KalmanFilter(transition_matrices=F, transition_covariance=Q, n_dim_obs = 3)
    #x_est, P_est = kf.filter(pos_obs)
    x_smooth, P_smooth = kf.smooth(pos_obs)
    #x_smooth, P_smooth = kf.em(pos_obs, n_iter = 30).smooth(pos_obs)
    return x_smooth, P_smooth


def run_test():
    Fs = 10
    t_end = 30.0

    dt = 1.0 / Fs
    sig_acc = np.diag([0.4, 0.01, 0.01]) # noise density of acceleration
    R = np.array([\
        [0.6, 0.0, 0.0],
        [0.0, 0.5, -0.0], \
        [0.0, -0.0, 0.2]]) # covariance of position observation

    x_init = np.array([0.0, 0.0, 0.0,0.0, 0.0, 0.0,0.0, 0.0, 0.0])
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

    pos_obs_file = 'pos3d_obs.csv'
    with open(pos_obs_file, 'w') as f:
        f.write('time,x,y,z,s_xx,s_yy,s_zz,s_xy,s_xz,s_yz\n')
        #for 
    print(pos_obs_file)

    x_est, P_est = run_pykalman_kalman_filter(dt, sig_acc, x_init, x_var_init, pos_obs, R)

    # ------
    fig = plot_kf_pva3d_states_filter(t_idx, x_est, x_true, pos_obs)
    ofile = 'kf_3d_state_filter_pykalman.png'
    plt.savefig(ofile)
    print(ofile)

    x_smooth, P_smooth = run_pykalman_kalman_smoother(dt, sig_acc, x_init, x_var_init, pos_obs, R)

    # ------
    fig = plot_kf_pva3d_states_smoother(t_idx, x_est, P_est, x_smooth, P_smooth, t_idx, x_true, pos_obs)
    ofile = 'kf_3d_state_smoother_pykalman.png'
    plt.savefig(ofile)
    print(ofile)

    # ------
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    plt.suptitle(f'Simulation')
    for _d, _lb in enumerate(['X', 'Y', 'Z']):
        axes[_d,0].plot(t_idx, [v[_d, _d] for v in P_est],  label='KF fliter')
        axes[_d,0].plot(t_idx, [v[_d, _d] for v in P_smooth],  label='KF smoothing')
    for _d, _lb in enumerate(['X', 'Y', 'Z']):
        axes[_d,1].plot(t_idx, [v[3+_d, 3+_d] for v in P_est],  label='KF fliter')
        axes[_d,1].plot(t_idx, [v[3+_d, 3+_d] for v in P_smooth],  label='KF smoothing')
    for _d, _lb in enumerate(['X', 'Y', 'Z']):
        axes[_d,2].plot(t_idx, [v[6+_d, 6+_d] for v in P_est],  label='KF fliter')
        axes[_d,2].plot(t_idx, [v[6+_d, 6+_d] for v in P_smooth],  label='KF smoothing')
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            axes[i,j].grid(True)
            axes[i,j].legend(loc='upper left')
            axes[i,j].set_yscale('log')
    #axes[0].set_ylabel(u'Var[$p(t)$] [m$^2$]')
    #axes[1].plot(t_est, vel_var_est,  label='KF fliter')
    #axes[1].plot(t_smooth, vel_var_smooth,  label='KF smoothing')
    #axes[1].set_ylabel(u'Var[$v(t)$] [m$^2$/s$^2$]')
    ofile = 'kf_3d_state_var_pykalman.png'
    plt.savefig(ofile)
    print(ofile)

if __name__ == '__main__':
    run_test()
