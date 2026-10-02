"""
Random acceleration model.
"""

import matplotlib.pyplot as plt
import numpy as np

sig0 = 0.15
Fs = 200
t_end = 3.0

pos_init, vel_init = 0.0, 0.0
dt = 1.0 / Fs

def get_stochastic_proc():
    stoproc = {'time':[], 'acc':[], 'vel': [], 'pos': []}
    v, p = vel_init, pos_init
    t = 0
    while True:
        _acc = sig0 * np.random.normal() / np.sqrt(dt)
        if t > t_end:
            break
        v = v + _acc * dt
        p = p + v * dt
        stoproc['time'].append(t)
        stoproc['acc'].append(_acc)
        stoproc['vel'].append(v)
        stoproc['pos'].append(p)
        t = t + dt
    return stoproc

N = 200
sim_data = []
for _ in range(N):
    _p = get_stochastic_proc()
    sim_data.append(_p)


def plot_posvel(sim_data, n_show=11, highlight=0):
    """Plot the first n_show sample paths: one highlighted in red, the rest in black."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 6))
    # Analytic standard deviations: acc = sigma/sqrt(dt) (white), Var[v] = sigma^2 t, Var[p] = sigma^2 t^3 / 3.
    t = np.array(sim_data[0]['time'])
    stds = {
        'acc': np.full_like(t, sig0 / np.sqrt(dt)),
        'vel': sig0 * np.sqrt(t),
        'pos': sig0 * np.sqrt(np.power(t, 3) / 3.0),
    }
    for a, key in zip(axes, ['acc', 'vel', 'pos']):
        a.fill_between(t, -2 * stds[key], 2 * stds[key], color='tab:blue', alpha=0.2, lw=0, label=r'$\pm 2\sigma$')
        for _i, _v in enumerate(sim_data[:n_show]):
            if _i != highlight:
                a.plot(_v['time'], _v[key], color='k', lw=0.6, alpha=0.5)
        # Draw the highlighted path last so it sits on top of the black ones.
        a.plot(sim_data[highlight]['time'], sim_data[highlight][key], color='r', lw=1.2)
    axes[0].set_ylabel('acceleration [m/s$^2$]')
    axes[1].set_ylabel('velocity[m/s]')

    for a, key in zip(axes, ['acc', 'vel', 'pos']):
        a.grid(True)
        a.set_xlim([0, t_end])
        # Symmetric y range about zero, sized to the largest excursion drawn in this panel (paths or 2-sigma band).
        ymax = max(max(np.max(np.abs(line.get_ydata())) for line in a.get_lines()), 2 * np.max(stds[key]))
        a.set_ylim([-1.05 * ymax, 1.05 * ymax])
    axes[0].legend(loc='upper right')
    axes[2].set_xlabel('time [s]')
    axes[2].set_ylabel('position[m]')
    fig.suptitle(f'Random acceleration model: {n_show} of N={N} sample paths shown (red: sample #{highlight})\n'
                 rf'noise density $\sigma$ = {sig0} m/s$^{{1.5}}$ (m/s$^2/\sqrt{{\mathrm{{Hz}}}}$), $F_s$ = {Fs} Hz')
    plt.tight_layout()
    plt.savefig('pos_vel_acc2.png')

plot_posvel(sim_data)


tlen = len(sim_data[0]['time'])
t_indices = []
vel_sample_vars = []
pos_sample_vars = []
velpos_sample_covs = []
for tidx in range(tlen):
    t = sim_data[0]['time'][tidx]
    t_indices.append(t)
    acc_samples = [v['acc'][tidx] for v in sim_data]
    vel_samples = [v['vel'][tidx] for v in sim_data]
    pos_samples = [v['pos'][tidx] for v in sim_data]
    #vel_sample_vars.append(np.var(vel_samples))
    #pos_sample_vars.append(np.var(pos_samples))
    cov_vp = np.cov(np.array([vel_samples, pos_samples]))
    #print(cov_vp)
    assert cov_vp.shape == (2,2)
    vel_sample_vars.append(cov_vp[0][0])
    pos_sample_vars.append(cov_vp[1][1])
    velpos_sample_covs.append(cov_vp[0][1])

fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
fig.suptitle(f'Random acceleration model: sample variance/covariance over N={N} sample paths\n'
             rf'noise density $\sigma$ = {sig0} m/s$^{{1.5}}$ (m/s$^2/\sqrt{{\mathrm{{Hz}}}}$), $F_s$ = {Fs} Hz')
t_indices = np.array(t_indices)
axes[0].set_title(r'Variance of velocity: $\mathrm{Var}[v(t)] = \sigma^2 t$')
axes[0].plot(t_indices, vel_sample_vars, label='sim')
axes[0].plot(t_indices, sig0 * sig0 * t_indices, label=r'$\sigma^2 t$')
axes[0].set_ylabel(r'$\mathrm{Var}[v(t)]$ [m$^2$/s$^2$]')

axes[1].set_title(r'Variance of position: $\mathrm{Var}[p(t)] = \frac{1}{3}\sigma^2 t^3$')
axes[1].plot(t_indices, pos_sample_vars, label='sim')
axes[1].plot(t_indices, (1.0/3.0) * np.power(sig0, 2) * np.power(t_indices, 3), label=r'$\frac{1}{3}\sigma^2 t^3$')
axes[1].set_ylabel(r'$\mathrm{Var}[p(t)]$ [m$^2$]')

axes[2].set_title(r'Covariance of position and velocity: $\mathrm{Cov}[p(t), v(t)] = \frac{1}{2}\sigma^2 t^2$')
axes[2].plot(t_indices, velpos_sample_covs, label='sim')
axes[2].plot(t_indices, (1.0/2.0) * np.power(sig0, 2) * np.power(t_indices, 2), label=r'$\frac{1}{2}\sigma^2 t^2$')
axes[2].set_ylabel(r'$\mathrm{Cov}[p(t), v(t)]$ [m$^2$/s]')
for a in axes:
    a.grid(True)
    a.legend(loc='upper left')
    a.margins(x=0)
#axes[1].set_yscale('log')
axes[2].set_xlabel('time [s]')
axes[0].set_xlim((0, t_end))
plt.savefig('vel_sample_var.png', bbox_inches='tight', pad_inches=0.05)
