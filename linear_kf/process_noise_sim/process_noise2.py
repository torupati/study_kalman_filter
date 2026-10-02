"""
Random acceleration model.
"""

import matplotlib.pyplot as plt
import numpy as np

sig0 = 0.15
Fs = 200
t_end = 3.0

pos_init, vel_init, acc_init = 0.0, 0.0, 0.0
dt = 1.0 / Fs

def get_stochastic_proc():
    stoproc = {'time':[], 'jerk': [], 'acc':[], 'vel': [], 'pos': []}
    a, v, p = acc_init, vel_init, pos_init
    t = 0
    while True:
        _jerk = sig0 * np.random.normal() * np.sqrt(dt)
        if t > t_end:
            break
        a = a + _jerk
        v = v + a * dt
        p = p + v * dt + 0.5 * a * dt * dt
        stoproc['time'].append(t)
        stoproc['jerk'].append(_jerk)
        stoproc['acc'].append(a)
        stoproc['vel'].append(v)
        stoproc['pos'].append(p)
        t = t + dt
    return stoproc

N = 500
sim_data = []
for _ in range(N):
    _p = get_stochastic_proc()
    sim_data.append(_p)


def plot_posvel(sim_data, n_show=11, highlight=0):
    """Plot the first n_show sample paths: one highlighted in red, the rest in black."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 6))
    # Analytic standard deviations: Var[a] = sigma^2 t, Var[v] = sigma^2 t^3 / 3, Var[p] = sigma^2 t^5 / 20.
    t = np.array(sim_data[0]['time'])
    stds = {
        'acc': sig0 * np.sqrt(t),
        'vel': sig0 * np.sqrt(np.power(t, 3) / 3.0),
        'pos': sig0 * np.sqrt(np.power(t, 5) / 20.0),
    }
    for a, key in zip(axes, ['acc', 'vel', 'pos']):
        a.fill_between(t, -2 * stds[key], 2 * stds[key], color='tab:blue', alpha=0.2, lw=0, label=r'$\pm 2\sigma$')
        for _i, _v in enumerate(sim_data[:n_show]):
            if _i != highlight:
                a.plot(_v['time'], _v[key], color='k', lw=0.6, alpha=0.5)
        # Draw the highlighted path last so it sits on top of the black ones.
        a.plot(sim_data[highlight]['time'], sim_data[highlight][key], color='r', lw=1.2)
    axes[0].set_ylabel('$a(t)$ [m/s$^2$]')
    axes[1].set_ylabel('$v(t)$ [m/s]')
    axes[2].set_ylabel('$p(t)$ [m]')

    for a, key in zip(axes, ['acc', 'vel', 'pos']):
        a.grid(True)
        a.set_xlim([0, t_end])
        # Symmetric y range about zero, sized to the largest excursion drawn in this panel (paths or 2-sigma band).
        ymax = max(max(np.max(np.abs(line.get_ydata())) for line in a.get_lines()), 2 * np.max(stds[key]))
        a.set_ylim([-1.05 * ymax, 1.05 * ymax])
    axes[0].legend(loc='upper right')
    axes[2].set_xlabel('time $t$ [s]')
    fig.suptitle(f'Random jerk model: {n_show} of N={N} sample paths shown (red: sample #{highlight})\n'
                 rf'noise density $\sigma$ = {sig0} m/s$^{{2.5}}$ (m/s$^3/\sqrt{{\mathrm{{Hz}}}}$), $F_s$ = {Fs} Hz')
    plt.tight_layout()
    plt.savefig('pos_vel_acc_model2.png')

plot_posvel(sim_data)


tlen = len(sim_data[0]['time'])
t_indices = []
acc_sample_vars = []
vel_sample_vars = []
pos_sample_vars = []
accvel_sample_covs = []
velpos_sample_covs = []
accpos_sample_covs = []
for tidx in range(tlen):
    t = sim_data[0]['time'][tidx]
    t_indices.append(t)
    acc_samples = [v['acc'][tidx] for v in sim_data]
    vel_samples = [v['vel'][tidx] for v in sim_data]
    pos_samples = [v['pos'][tidx] for v in sim_data]
    cov_vp = np.cov(np.array([acc_samples, vel_samples, pos_samples]))
    #print(cov_vp)
    assert cov_vp.shape == (3,3)
    acc_sample_vars.append(cov_vp[0][0])
    vel_sample_vars.append(cov_vp[1][1])
    pos_sample_vars.append(cov_vp[2][2])
    accvel_sample_covs.append(cov_vp[0][1])
    velpos_sample_covs.append(cov_vp[1][2])
    accpos_sample_covs.append(cov_vp[0][2])

fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
fig.suptitle(f'Random jerk model: sample variance/covariance over N={N} sample paths\n'
             rf'noise density $\sigma$ = {sig0} m/s$^{{2.5}}$ (m/s$^3/\sqrt{{\mathrm{{Hz}}}}$), $F_s$ = {Fs} Hz')
t_indices = np.array(t_indices)
axes[0][0].set_title('variance of $a(t)$')
axes[0][0].plot(t_indices, acc_sample_vars, label='sim')
axes[0][0].plot(t_indices, sig0 * sig0 * t_indices, label=r'$\sigma^2 t$')

axes[1][0].set_title('variance of $v(t)$')
axes[1][0].plot(t_indices, vel_sample_vars, label='sim')
axes[1][0].plot(t_indices, (1.0/3.0) * np.power(sig0, 2) * np.power(t_indices, 3), label=r'$1/3\sigma^2 t^3$')

axes[2][0].set_title('variance of $p(t)$')
axes[2][0].plot(t_indices, pos_sample_vars, label='sim')
axes[2][0].plot(t_indices, (1.0/20.0) * np.power(sig0, 2) * np.power(t_indices, 5), label=r'$1/20\sigma^2 t^5$')

axes[0][1].set_title('covariance of $a(t)$ and $v(t)$')
axes[0][1].plot(t_indices, accvel_sample_covs, label='sim')
axes[0][1].plot(t_indices, (1.0/2.0) * np.power(sig0, 2) * np.power(t_indices, 2), label=r'$1/2\sigma^2 t^2$')

axes[1][1].set_title('covariance of $v(t)$ and $p(t)$')
axes[1][1].plot(t_indices, velpos_sample_covs, label='sim')
axes[1][1].plot(t_indices, (1.0/8.0) * np.power(sig0, 2) * np.power(t_indices, 4), label=r'$1/8\sigma^2 t^4$')

axes[2][1].set_title('covariance of $a(t)$ and $p(t)$')
axes[2][1].plot(t_indices, accpos_sample_covs, label='sim')
axes[2][1].plot(t_indices, (1.0/6.0) * np.power(sig0, 2) * np.power(t_indices, 3), label=r'$1/6\sigma^2 t^3$')

for i in range(axes.shape[0]):
    for j in range(axes.shape[1]):
        axes[i,j].grid(True)
        axes[i,j].legend()
#axes[1].set_yscale('log')
axes[2][0].set_xlabel('time [s]')
axes[2][1].set_xlabel('time [s]')
plt.tight_layout()
plt.savefig('model2_cov_sim.png')

