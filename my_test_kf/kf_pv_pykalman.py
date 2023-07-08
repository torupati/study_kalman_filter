import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
# Plot error bars in estimation

Fs = 10
#t_motion_start, t_motion_end = 1.0, 9.0
t_end = 10.0

sig0 = 2.0 # 1.0 # noise density of acceleration
sig1 = 0.5  # standard deviation of position observation

H = np.array([1, 0]).reshape(1, 2)
R = [sig1 * sig1]

dt = 1.0/Fs
x_var_init = np.array([[10.0, 0.0], [0.0, 10.0]])
# Generate true and observed position
t_idx = np.arange(0.0, t_end, dt)
W = 4.0 * np.pi / t_end
pos_true = 2.3 * (np.cos(W * t_idx) + 0.1)
vel_true = 2.3 * W * (-np.sin(W * t_idx))
pos_obs = pos_true + np.random.normal(0.0, sig1*sig1, len(pos_true))

# ---
# Kalman Filter
x = np.array([0.0, 0.0]).reshape(2, 1)
P = x_var_init
F = np.array([\
        [1.0, dt],
        [0.0, 1.0]])
Q = np.array([\
        [1.0 / 3.0 * np.power(sig0, 2) * np.power(dt, 3), 1.0 / 2.0 * np.power(sig0, 2) * np.power(dt, 2)], \
        [1.0 / 2.0 * np.power(sig0, 2) * np.power(dt, 2), np.power(sig0, 2)* dt]])

kf = KalmanFilter(transition_matrices=F,
                transition_covariance=Q)

x_est, P_est = kf.filter(pos_obs)
#
fig, axes = plt.subplots(2, 1)
plt.suptitle(f'Simulation (acc noise={sig0:.4f})')
axes[0].plot(t_idx, x_est[:,0],  label='KF fliter')
axes[0].plot(t_idx, pos_true, label='true')
axes[0].plot(t_idx, pos_obs, '.', label="obs")
axes[0].set_ylabel('position [m]')
axes[1].plot(t_idx, x_est[:,1],  label='KF fliter')
axes[1].plot(t_idx, vel_true, label='true')
axes[1].set_ylabel('velocity [m/s]')
for a in axes:
    a.grid(True)
    a.legend()

ofile = 'kf_1d_state_filter_pykalman.png'
plt.savefig(ofile)
print(ofile)
# ---
x_smooth, P_smooth = kf.smooth(pos_obs)

#---
fig, axes = plt.subplots(2, 1)
plt.suptitle(f'Simulation (acc noise={sig0:.4f})')
axes[0].plot(t_idx, x_est[:,0],  label='KF fliter')
axes[0].plot(t_idx, x_smooth[:,0],  label='KF smoothing')
axes[0].plot(t_idx, pos_true, label='true')
axes[0].plot(t_idx, pos_obs, '.', label="obs")
axes[0].set_ylabel('position [m]')
axes[1].plot(t_idx, x_est[:,1],  label='KF fliter')
axes[1].plot(t_idx, x_smooth[:,1],  label='KF smoothing')
axes[1].plot(t_idx, vel_true, label='true')
axes[1].set_ylabel('velocity [m/s]')
for a in axes:
    a.grid(True)
    a.legend()

ofile = 'kf_1d_state_smooth_pykalman.png'
plt.savefig(ofile)
print(ofile)

#
fig, axes = plt.subplots(2, 1)
plt.suptitle(f'Simulation (acc noise={sig0:.4f})')
axes[0].plot(t_idx, P_est[:,0,0],  label='KF fliter')
axes[0].plot(t_idx, P_smooth[:,0,0],  label='KF smoothing')
axes[0].set_ylabel(u'Var[$p(t)$] [m$^2$]')
axes[1].plot(t_idx, P_est[:,1,1], label='KF fliter')
axes[1].plot(t_idx, P_smooth[:,1,1],  label='KF smoothing')
axes[1].set_ylabel(u'Var[$v(t)$] [m$^2$/s$^2$]')
for a in axes:
    a.grid(True)
    a.set_yscale('log')
    a.legend()

plt.savefig('kf_1d_state_var_pykalman.png')

