import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter

from x_generator import generte_true_pos_vel_1d_type1, kf_pv_1d_type1_condition
from kf_pv_plot import plot_kf_1d_var

# Plot error bars in estimation
Fs, t_end, sig0, sig1, x_init, x_var_init = kf_pv_1d_type1_condition()

# Generate true and observed position
t_idx = np.arange(0.0, t_end, 1.0/Fs)
x_true = generte_true_pos_vel_1d_type1(t_idx, t_end/2.0)
pos_obs = np.array(x_true)[:,0] + np.random.normal(0.0, sig1*sig1, len(x_true))

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

#kf = KalmanFilter(transition_matrices=F,
#                transition_covariance=Q)
print(f'F {F.shape} H {H.shape}  Q {Q.shape} R {R.shape} x0 {x_init.shape} x0_cov {x_var_init.shape}')

kf = KalmanFilter(transition_matrices = F, 
                  observation_matrices = H, 
                  transition_covariance = Q, 
                  observation_covariance = R,
                  initial_state_covariance = x_var_init,
                  initial_state_mean = x_init)

# Run Kalman filter
x_est, P_est = kf.filter(pos_obs)

#
fig, axes = plt.subplots(2, 1)
plt.suptitle(f'Simulation (acc noise={sig0:.4f})')
axes[0].plot(t_idx, x_est[:,0],  label='KF fliter')
axes[0].plot(t_idx, [_v[0] for _v in x_true], label='true')
axes[0].plot(t_idx, pos_obs, '.', label="obs")
axes[0].set_ylabel('position [m]')
axes[1].plot(t_idx, x_est[:,1],  label='KF fliter')
axes[1].plot(t_idx, [_v[1] for _v in x_true], label='true')
axes[1].set_ylabel('velocity [m/s]')
for a in axes:
    a.grid(True)
    a.legend()
ofile = 'kf_1d_state_filter_pykalman.png'
fig.savefig(ofile)
print(ofile)

# ---
x_smooth, P_smooth = kf.smooth(pos_obs)

#---
fig, axes = plt.subplots(2, 1)
plt.suptitle(f'Simulation (acc noise={sig0:.4f})')
axes[0].plot(t_idx, x_est[:,0],  label='KF fliter')
axes[0].plot(t_idx, x_smooth[:,0],  label='KF smoothing')
axes[0].plot(t_idx, [_v[0] for _v in x_true], label='true')
axes[0].plot(t_idx, pos_obs, '.', label="obs")
axes[0].set_ylabel('position [m]')
axes[1].plot(t_idx, x_est[:,1],  label='KF fliter')
axes[1].plot(t_idx, x_smooth[:,1],  label='KF smoothing')
axes[1].plot(t_idx, [ _v[1] for _v in x_true], label='true')
axes[1].set_ylabel('velocity [m/s]')
for a in axes:
    a.grid(True)
    a.legend()

ofile = 'kf_1d_state_smooth_pykalman.png'
fig.savefig(ofile)
print(ofile)

# Plot vovariance matrix
ofile = 'kf_1d_state_var_pykalman.png'
title_str = f'Simulation (acc noise={sig0:.4f})'
fig = plot_kf_1d_var(t_idx, P_est, P_smooth, title_str)
fig.savefig(ofile)
print(ofile)
