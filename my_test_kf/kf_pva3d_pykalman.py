import numpy as np
import json
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

def save_params(fname, sig_acc, x_init, x_var_init):
    """
    """
    if type(sig_acc) == np.ndarray:
        sig_acc = sig_acc.tolist()
    if type(x_init) == np.ndarray:
        x_init = x_init.tolist()
    if type(x_var_init) == np.ndarray:
        x_var_init = x_var_init.tolist()

    _out = {\
        'name': 'kf_pva3d',
        'parameters': {'sig_acc': sig_acc, 'x_init': x_init, 'x_var_init': x_var_init}
    }
    with open(fname, 'w') as _f:
        json.dump(_out, _f, indent=2)

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

    save_params('param_kf_pva3d.json', sig_acc, x_init, x_var_init)


    # Generate true and observed position
    t_idx = np.arange(0.0, t_end, 1.0/Fs)
    x_true = generate_true_pos_vel_acc_3d_type1(t_idx)
    pos_obs = np.array([[_x[0], 0, 0] + np.random.multivariate_normal(np.zeros(3), R) for _x in x_true])

    pos_obs_file = 'pos3d_obs.csv'
    with open(pos_obs_file, 'w') as f:
        f.write('time,x,y,z,s_xx,s_yy,s_zz,s_xy,s_xz,s_yz\n')
        for _t, _v in zip(t_idx, pos_obs):
            f.write(f'{_t:.3f},{_v[0]:.3f},{_v[1]:.3f},{_v[2]:.3f}')
            s_xx, s_yy, s_zz = R[0, 0], R[1, 1], R[2, 2]
            s_xy, s_yz, s_xz = R[0, 1], R[1, 2], R[0, 2]
            f.write(f',{s_xx:.3f},{s_yy:.3f},{s_zz:.3f},{s_xy:.3f},{s_yz:.3f},{s_xz:.3f}')
            f.write('\n')
        print(f.name)

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

#run_test()


def main(args):
    # load observation
    try:
        data = np.loadtxt(args.obs_file, skiprows=1, delimiter=',')
    except Exception as e:
        print(e)
        return
    t_idx = data[:, 0]
    dt = t_idx[1] - t_idx[0]
    pos_obs = data[:, 1:1+3]
    pos_var = data[0, 4:4 + 6]
    s_xx, s_yy, s_zz, s_xy, s_xz, s_yz = pos_var[0], pos_var[1], pos_var[2], pos_var[3], pos_var[4], pos_var[5]
    R = np.array([[s_xx, s_xy, s_xz], [s_xy, s_yy, s_yz], [s_xz, s_yz, s_zz]])

    with open(args.param_file) as f:
        prm = json.load(f)
        print(prm)
        sig_acc = np.array(prm['parameters']['sig_acc'])
        x_init = prm['parameters']['x_init']
        x_var_init = prm['parameters']['x_var_init']
    print('dt=', dt)
    print('sig_acc=',sig_acc)

    x_est, P_est = run_pykalman_kalman_filter(dt, sig_acc, x_init, x_var_init, pos_obs, R)
    x_smooth, P_smooth = run_pykalman_kalman_smoother(dt, sig_acc, x_init, x_var_init, pos_obs, R)

    # ------
    fig = plot_kf_pva3d_states_smoother(t_idx, x_est, P_est, x_smooth, P_smooth, t_idx, [], pos_obs)
    ofile = 'out_kf_3d_state_smoother_pykalman.png'
    plt.savefig(ofile)
    print(ofile)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
                    prog='kf_pva3d_pykalman',
                    description='Run offline estimation of Kalman Filter 3D PVA',
                    epilog='o_o....... torupati ......o_o')
    parser.add_argument('param_file', type=str, help='kf_pva3d parameter file (*.json)')
    parser.add_argument('obs_file', type=str, help='observation data (*.csv)')
    #parser.add_argument('-c', '--count')
    parser.add_argument('-d', '--debug', action='store_true', help='logging in debug mode')
    args = parser.parse_args()

    main(args)

