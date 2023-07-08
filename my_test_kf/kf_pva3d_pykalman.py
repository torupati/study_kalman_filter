import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter

Fs = 10
t_end = 30.0

dt = 1.0 / Fs
sig_acc = np.diag([0.4, 0.01, 0.01]) # noise density of acceleration
R = np.array([\
    [0.6, 0.0, 0.0],
    [0.0, 0.5, -0.0], \
    [0.0, -0.0, 0.2]]) # covariance of position observation

H = np.array([\
    [1, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 1, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 1, 0, 0, 0, 0, 0, 0]]) #.reshape(3, 9)

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

# Generate true and observed position
t_idx = np.arange(0.0, t_end, 1.0/Fs)
A, W = 5.0, 2.0 * np.pi / 10.0
pos_true = A * (np.cos(W * t_idx) - 1.0)
vel_true = A * (-W * np.sin(W * t_idx))
acc_true = A * (-W * W * np.sin(W * t_idx))
#pos_true, vel_true, acc_true = np.zeros(len(t_idx)), np.zeros(len(t_idx)), np.zeros(len(t_idx))
x_true = np.array([[_x, 0, 0, _v, 0, 0, _a, 0, 0] for _x, _v, _a in zip(pos_true, vel_true, acc_true)])
pos_obs = np.array([[_x, 0, 0] + np.random.multivariate_normal(np.zeros(3), R) for _x in pos_true])

# ---
# Kalman Filter
F = np.array([\
        [1, 0, 0, dt, 0, 0, 0.5*dt*dt, 0, 0], \
        [0, 1, 0, 0, dt, 0, 0, 0.5*dt*dt, 0], \
        [0, 0, 1, 0, 0, dt, 0, 0, 0.5*dt*dt], \
        [0, 0, 0, 1, 0, 0, dt, 0, 0], \
        [0, 0, 0, 0, 1, 0, 0, dt, 0], \
        [0, 0, 0, 0, 0, 1, 0, 0, dt], \
        [0, 0, 0, 0, 0, 0, 1, 0, 0], \
        [0, 0, 0, 0, 0, 0, 0, 1, 0], \
        [0, 0, 0, 0, 0, 0, 0, 0, 1]])
Q = np.zeros([9, 9])
Q[0:3, 0:3] = 1.0 / 20.0 * np.power(dt, 5) * sig_acc # E[p(t)p(t)]
Q[0:3, 3:6] = Q[3:6, 0:3] = 1.0 / 8.0 * np.power(dt, 4) * sig_acc # E[p(t)v(t)]
Q[0:3, 6:9] = Q[6:9, 0:3] =  np.power(dt, 3) * sig_acc * (1.0 / 6.0)# E[p(t)a(t)]
Q[3:6, 3:6] = 1.0 / 3.0 * np.power(dt, 3) * sig_acc # E[v(t)v(t)]
Q[3:6, 0:3] = Q[3:6, 0:3] = 1.0 / 2.0 * np.power(dt, 2) * sig_acc # E[v(t)a(t)]
Q[6:9, 6:9] = dt * sig_acc # E[a(t)a(t)]

# ------
kf = KalmanFilter(transition_matrices = F, 
                  observation_matrices = H, 
                  transition_covariance = Q, 
                  observation_covariance = R,
                  initial_state_mean = x_init,
                  initial_state_covariance = x_var_init)
x_est, P_est = kf.filter(pos_obs)

# ------
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

ofile = 'kf_3d_state_filter_pykalman.png'
plt.savefig(ofile)
print(ofile)

# ------
kf = KalmanFilter(transition_matrices=F, transition_covariance=Q, n_dim_obs = 3)
#x_smooth, P_smooth = kf.smooth(pos_obs)
x_smooth, P_smooth = kf.em(pos_obs, n_iter = 30).smooth(pos_obs)

# ------
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
plt.suptitle(f'Simulation')
for _d, _lb in enumerate(['X', 'Y', 'Z']):
    axes[_d,0].plot(t_idx, x_est[:,_d],  label=f'KF fliter')
    axes[_d,0].plot(t_idx, x_smooth[:,_d],  label=f'KF smoother')
    axes[_d,0].plot(t_idx, [v[_d] for v in x_true], label='true')
    axes[_d,0].plot(t_idx, [v[_d] for v in pos_obs], 'o', label="obs")
    axes[_d,0].set_ylabel(f'position {_lb} [m]')
for _d, _lb in enumerate(['X', 'Y', 'Z']):
    axes[_d,1].plot(t_idx, x_est[:,3+_d],  label=f'KF fliter')
    axes[_d,1].plot(t_idx, x_smooth[:,3+_d],  label=f'KF smoother')
    axes[_d,1].plot(t_idx, [v[3+_d] for v in x_true], label=f'true {_lb}')
    axes[_d,1].set_ylabel(f'Velocity {_lb} [m]')
for _d, _lb in enumerate(['X', 'Y', 'Z']):
    axes[_d,2].plot(t_idx, x_est[:,6+_d],  label=f'KF fliter')
    axes[_d,2].plot(t_idx, x_smooth[:,6+_d],  label=f'KF smoother')
    axes[_d,2].plot(t_idx, [v[6+_d] for v in x_true], label=f'true {_lb}')
    axes[_d,2].set_ylabel(u'acc [m/s$^2$]')
for i in range(axes.shape[0]):
    for j in range(axes.shape[1]):
        axes[i,j].grid(True)
        axes[i,j].legend(loc='upper left')

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
