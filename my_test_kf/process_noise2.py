"""
Random acceleration model.
"""

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

import matplotlib.pyplot as plt
def plot_posvel(sim_data):
    fig, axes = plt.subplots(3, 1, figsize=(10, 6))
    for _i, _v in enumerate(sim_data):
        if _i > 10:
            break
        axes[0].plot (_v['time'], _v['acc'])
#    axes[0].axhline(y=sig0*np.sqrt(dt), color='r')
#    axes[0].axhline(y=-sig0*np.sqrt(dt), color='r')
    axes[0].set_ylabel(u'$a(t)$ [m/s$^2$]')
    for _i, _v in enumerate(sim_data):
        if _i > 10:
            break
        axes[1].plot(_v['time'], _v['vel'])
    axes[1].set_ylabel(u'$v(t)$ [m/s]')
    dts = np.arange(0, t_end, 0.1)
 #   axes[1].plot(dts, sig0 * np.sqrt(dts), color='r')
 #   axes[1].plot(dts, -sig0 * np.sqrt(dts), color='r')
    for _i, _v in enumerate(sim_data):
        if _i > 10:
            break
        axes[2].plot(_v['time'], _v['pos'])
    axes[2].set_ylabel(u'$p(t)$ [m]')
#    axes[2].plot(dts, 1.0/3.0 * sig0 * np.power(dts, 1.5), color='r')
#    axes[2].plot(dts, -1.0/3.0 * sig0 * np.power(dts, 1.5), color='r')
    
    for a in axes:
        a.grid(True)
        a.set_xlim([0, t_end])
    axes[2].set_xlabel('time $t$ [s]')
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
fig.suptitle(f'Simulation of Random Jerk Model N={N}')
t_indices = np.array(t_indices)
axes[0][0].set_title(u'variance of $a(t)$')
axes[0][0].plot(t_indices, acc_sample_vars, label='sim')
axes[0][0].plot(t_indices, sig0 * sig0 * t_indices, label=u'$\sigma^2 t$')

axes[1][0].set_title(u'variance of $v(t)$')
axes[1][0].plot(t_indices, vel_sample_vars, label='sim')
axes[1][0].plot(t_indices, (1.0/3.0) * np.power(sig0, 2) * np.power(t_indices, 3), label=u'$1/3\sigma^2 t^3$')

axes[2][0].set_title(u'variance of $p(t)$')
axes[2][0].plot(t_indices, pos_sample_vars, label='sim')
axes[2][0].plot(t_indices, (1.0/20.0) * np.power(sig0, 2) * np.power(t_indices, 5), label=u'$1/20\sigma^2 t^5$')

axes[0][1].set_title(u'covariance of $a(t)$ and $v(t)$')
axes[0][1].plot(t_indices, accvel_sample_covs, label='sim')
axes[0][1].plot(t_indices, (1.0/2.0) * np.power(sig0, 2) * np.power(t_indices, 2), label=u'$1/2\sigma^2 t^2$')

axes[1][1].set_title(u'covariance of $v(t)$ and $p(t)$')
axes[1][1].plot(t_indices, velpos_sample_covs, label='sim')
axes[1][1].plot(t_indices, (1.0/8.0) * np.power(sig0, 2) * np.power(t_indices, 4), label=u'$1/8\sigma^2 t^4$')

axes[2][1].set_title(u'covariance of $a(t)$ and $p(t)$')
axes[2][1].plot(t_indices, accpos_sample_covs, label='sim')
axes[2][1].plot(t_indices, (1.0/6.0) * np.power(sig0, 2) * np.power(t_indices, 3), label=u'$1/6\sigma^2 t^3$')

for i in range(axes.shape[0]):
    for j in range(axes.shape[1]):
        axes[i,j].grid(True)
        axes[i,j].legend()
#axes[1].set_yscale('log')
axes[2][0].set_xlabel('time [s]')
axes[2][1].set_xlabel('time [s]')
plt.tight_layout()
plt.savefig('model2_cov_sim.png')

