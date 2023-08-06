import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter

from x_generator import generte_true_pos_vel_1d_type1, kf_pv_1d_type1_condition
from kf_pv_plot import plot_kf_1d_var

np.random.seed(0)

# Plot error bars in estimation
Fs, t_end, sig0, sig1, x_init, x_var_init = kf_pv_1d_type1_condition()

# Generate true and observed position
t_idx = np.arange(0.0, t_end, 1.0/Fs)
x_true = generte_true_pos_vel_1d_type1(t_idx, t_end/2.0)
pos_obs = np.array(x_true)[:,0] + np.random.normal(0.0, sig1*sig1, len(x_true))

n_iter = 300 # EM iteration steps

# ---
# Kalman Filter
dt = 1.0/Fs
F = np.array([\
        [1.0, dt],
        [0.0, 1.0]])
H = np.array([1, 0]).reshape(1, 2)
Q = np.array([\
        [1.0 / 3.0 * np.power(sig0, 2) * np.power(dt, 3), 1.0 / 2.0 * np.power(sig0, 2) * np.power(dt, 2)], \
        [1.0 / 2.0 * np.power(sig0, 2) * np.power(dt, 2), np.power(sig0, 2)* dt]])
R = np.array([sig1 * sig1]).reshape(1,1)

x_init = x_init + np.array([-10,0])
kf = KalmanFilter(transition_matrices = F, 
                  observation_matrices = H, 
                  transition_covariance = Q, 
                  observation_covariance = R,
                  initial_state_covariance = x_var_init,
                  initial_state_mean = x_init,
                  em_vars = ['initial_state_mean'])

#kf = KalmanFilter(transition_matrices=F, transition_covariance=Q, \
#    em_vars = ['transition_covariance'])
#    em_vars = ['transition_covariance', 'observation_covariance'])

# ---
print('before learning: x_init=', kf.initial_state_mean)
x_est, P_est = kf.smooth(pos_obs)
#for i in range(100):
#    pos_obs = np.array(x_true)[:,0] + np.random.normal(0.0, sig1*sig1, len(x_true))
#    x_em, P_em = kf.em(pos_obs, n_iter = 1).smooth(pos_obs)
x_em, P_em = kf.em(pos_obs, n_iter = n_iter).smooth(pos_obs)
print('after learning: x_init', kf.initial_state_mean)

#---
fig, axes = plt.subplots(2, 1)
plt.suptitle(u'Simulation ($\sigma^2$={a})'.format(a=sig0))
axes[0].plot(t_idx, x_est[:,0],  label='Kalman smoother')
axes[0].plot(t_idx, x_em[:,0], label=u'Kalman smoother EM(itr={n})'.format(n=n_iter))
axes[0].plot(t_idx,[v[0] for v in x_true], label='true')
axes[0].plot(t_idx, pos_obs, '.', label="obs")
axes[0].set_ylabel('position [m]')
axes[1].plot(t_idx, x_est[:,1],  label='KF smoother')
axes[1].plot(t_idx, x_em[:,1], label=u'Kalman smoother EM(itr={n})'.format(n=n_iter))
axes[1].plot(t_idx, [v[1] for v in x_true], label='true')
axes[1].set_ylabel('velocity [m/s]')
for a in axes:
    a.grid(True)
    a.legend()

ofile = 'kf_1d_em_pykalman.png'
plt.savefig(ofile)
print(ofile)

#
fig, axes = plt.subplots(2, 1)
plt.suptitle(u'Simulation ($\sigma^2$={a})'.format(a=sig0))
#axes[0].plot(t_idx, P_est[:,0,0],  label='KF fliter')
axes[0].plot(t_idx, P_em[:,0,0],  label='KF smoothing')
axes[0].set_ylabel(u'Var[$p(t)$] [m$^2$]')
#axes[1].plot(t_idx, P_est[:,1,1], label='KF fliter')
axes[1].plot(t_idx, P_em[:,1,1],  label='KF smoothing')
axes[1].set_ylabel(u'Var[$v(t)$] [m$^2$/s$^2$]')
for a in axes:
    a.grid(True)
    a.set_yscale('log')
    a.legend()

plt.savefig('kf_1d_state_var_pykalman.png')

#x_smooth, P_smooth = kf.em(pos_obs, n_iter = 100).smooth(pos_obs)
