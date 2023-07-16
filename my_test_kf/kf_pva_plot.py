import matplotlib.pyplot as plt

def plot_kf_pva3d_states_filter(t_idx, x_est, x_true, pos_obs):

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    plt.suptitle(f'Simulation')
    for _d, _lb in enumerate(['X', 'Y', 'Z']):
        axes[_d,0].plot(t_idx, [v[_d] for v in x_true], label='true')
        axes[_d,0].plot(t_idx, x_est[:,_d], '.', label=f'KF fliter {_lb}')
        axes[_d,0].plot(t_idx, pos_obs[:,_d], 'o', label="obs")
        axes[_d,0].set_ylabel(f'position {_lb} [m]')
    for _d, _lb in enumerate(['X', 'Y', 'Z']):
        axes[_d,1].plot(t_idx, [v[3+_d] for v in x_true], label=f'true {_lb}')
        axes[_d,1].plot(t_idx, x_est[:,3+_d], '.', label=f'KF fliter {_lb}')
        axes[_d,1].set_ylabel(f'Velocity {_lb} [m]')
    for _d, _lb in enumerate(['X', 'Y', 'Z']):
        axes[_d,2].plot(t_idx, [v[6+_d] for v in x_true], label=f'true')
        axes[_d,2].plot(t_idx, x_est[:,6+_d], '.', label='KF fliter')
        axes[_d,2].set_ylabel(u'acc [m/s$^2$]')
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            axes[i,j].grid(True)
            axes[i,j].legend(loc='upper left')
    return fig


def plot_kf_pva3d_states_smoother(t_est, x_est, P_est, x_smooth, P_smooth, t_idx, x_true, pos_obs):
    """
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    plt.suptitle(f'Simulation')
    for _d, _lb in enumerate(['X', 'Y', 'Z']):
        axes[_d,0].plot(t_est, [v[_d] for v in x_est],  label=f'KF fliter')
        axes[_d,0].plot(t_est, [v[_d] for v in x_smooth],  label=f'KF smoother')
        axes[_d,0].plot(t_idx, [v[_d] for v in x_true], label='true')
        axes[_d,0].plot(t_idx, [v[_d] for v in pos_obs], 'o', label="obs")
        axes[_d,0].set_ylabel(f'position {_lb} [m]')
    for _d, _lb in enumerate(['X', 'Y', 'Z']):
        axes[_d,1].plot(t_est, [v[3+_d] for v in x_est],  label=f'KF fliter')
        axes[_d,1].plot(t_est, [v[3+_d] for v in x_smooth],  label=f'KF smoother')
        axes[_d,1].plot(t_idx, [v[3+_d] for v in x_true], label=f'true {_lb}')
        axes[_d,1].set_ylabel(f'Velocity {_lb} [m]')
    for _d, _lb in enumerate(['X', 'Y', 'Z']):
        axes[_d,2].plot(t_est, [v[6+_d] for v in x_est],  label=f'KF fliter')
        axes[_d,2].plot(t_est, [v[6+_d] for v in x_smooth],  label=f'KF smoother')
        axes[_d,2].plot(t_idx, [v[6+_d] for v in x_true], label=f'true {_lb}')
        axes[_d,2].set_ylabel(u'acc [m/s$^2$]')
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            axes[i,j].grid(True)
            axes[i,j].legend(loc='upper left')
    return fig

