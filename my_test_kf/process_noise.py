"""
Random acceleration model.
"""

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

import matplotlib.pyplot as plt
def plot_posvel(sim_data):
    fig, axes = plt.subplots(3, 1, figsize=(10, 6))
    for _i, _v in enumerate(sim_data):
        if _i > 10:
            break
        axes[0].plot (_v['time'], _v['acc'])
    axes[0].axhline(y=sig0/np.sqrt(dt), color='r')
    axes[0].axhline(y=-sig0/np.sqrt(dt), color='r')
    axes[0].set_ylabel(u'acceleration [m/s$^2$]')
    for _i, _v in enumerate(sim_data):
        if _i > 10:
            break
        axes[1].plot(_v['time'], _v['vel'])
    dts = np.arange(0, t_end, 0.1)
    #axes[1].plot(dts, sig0 * np.sqrt(dts), color='r')
    #axes[1].plot(dts, -sig0 * np.sqrt(dts), color='r')
    axes[1].set_ylabel('velocity[m/s]')
    for _i, _v in enumerate(sim_data):
        if _i > 10:
            break
        axes[2].plot (_v['time'], _v['pos'])
    #axes[2].plot(dts, 1.0/3.0 * sig0 * np.power(dts, 1.5), color='r')
    #axes[2].plot(dts, -1.0/3.0 * sig0 * np.power(dts, 1.5), color='r')
    
    for a in axes:
        a.grid(True)
        a.set_xlim([0, t_end])
    axes[2].set_xlabel('time [s]')
    axes[2].set_ylabel('position[m]')
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

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
t_indices = np.array(t_indices)
axes[0].set_title('variance of velocity')
axes[0].plot(t_indices, vel_sample_vars, label='sim')
axes[0].plot(t_indices, sig0 * sig0 * t_indices, label=u'$\sigma^2 t$')
axes[0].set_ylabel(u'$Var[v(t)]$ [m/s]')

axes[1].set_title('variance of position')
axes[1].plot(t_indices, pos_sample_vars, label='sim')
axes[1].plot(t_indices, (1.0/3.0) * np.power(sig0, 2) * np.power(t_indices, 3), label=u'$1/3\sigma^2 t^3$')
axes[1].set_ylabel(u'$Var[p(t)]$ [m]')

axes[2].set_title('covariance of position and velocity')
axes[2].plot(t_indices, velpos_sample_covs, label='sim')
axes[2].plot(t_indices, (1.0/2.0) * np.power(sig0, 2) * np.power(t_indices, 2), label=u'$1/2\sigma^2 t^3$')
axes[2].set_ylabel(u'$Cov[p(t)v(t)]$ [m$^2$/s]')
for a in axes:
    a.grid(True)
    a.legend()
#axes[1].set_yscale('log')
axes[2].set_xlabel('time [s]')
axes[0].set_xlim((0, t_end))
fig.subplots_adjust(hspace=0.4)
plt.savefig('vel_sample_var.png')

