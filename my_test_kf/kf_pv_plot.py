import matplotlib.pyplot as plt

def plot_kf_1d_var(t_idx, P_est, P_smooth, title_str: str = ''):
    """
    Plot Covariance sequence(length N) of position-velocity state space.

    Args:
    t_idx: time sequence (shape is (N, ))
    P_est, P_smooth: covariance sequence (shape is (N, 2, 2))

    Returns:
    figure
    """
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    plt.suptitle(title_str)
    axes[0].plot(t_idx, P_est[:, 0, 0], label='KF fliter')
    axes[0].plot(t_idx, P_smooth[:, 0, 0], label='KF smoothing')
    axes[0].set_ylabel(u'Var[$p(t)$] [m$^2$]')
    axes[0].set_yscale('log')
    axes[1].plot(t_idx, P_est[:, 1, 1], label='KF fliter')
    axes[1].plot(t_idx, P_smooth[:, 1, 1], label='KF smoothing')
    axes[1].set_ylabel(u'Var[$v(t)$] [m$^2$/s$^2$]')
    axes[1].set_yscale('log')
    axes[2].plot(t_idx, P_est[:, 0, 1], label='KF fliter')
    axes[2].plot(t_idx, P_smooth[:, 0, 1], label='KF smoothing')
    axes[2].set_ylabel(u'Cov[$p(t)v(t)$] [m$^2$/s]')
    for a in axes:
        a.grid(True)
        a.legend(loc='upper right')
    axes[2].set_xlim([t_idx[0], t_idx[-1]])
    axes[2].set_xlabel('time [s]')
    return fig
