import numpy as np
import matplotlib.pyplot as plt

from x_generator import generte_true_pos_vel_1d_type1, kf_pv_1d_type1_condition
from kf_pv_plot import plot_kf_1d_var, plot_kf_1d_filter_smooth

# Plot error bars in estimation
Fs, t_end, sig0, sig1, x_init, x_var_init = kf_pv_1d_type1_condition()

R = [sig1 * sig1]
# Generate true and observed position
t_idx = np.arange(0.0, t_end, 1.0/Fs)
x_true = generte_true_pos_vel_1d_type1(t_idx, t_end/2.0)
pos_obs = np.array(x_true)[:,0] + np.random.normal(0.0, sig1*sig1, len(x_true))

H = np.array([1, 0]).reshape(1, 2)

fig, axes = plt.subplots(2, 1)
axes[0].plot(t_idx, [v[0] for v in x_true],  label='true')
axes[0].plot(t_idx, pos_obs, '.', label="obs")
axes[1].plot(t_idx, [v[1] for v in x_true], label='true')
for a in axes:
    a.grid(True)
    a.legend()

plt.savefig('posvel_kf_pv_1d.png')

# Kalman Filter
x = x_init.reshape(2,1)
P = x_var_init
t_est = []
x_est = []
P_est = []
t_prev = 0.0
for _i, (t, y) in enumerate(zip(t_idx, pos_obs)):
    # measurement update
    S = np.dot(np.dot(H, P), H.T) + R  # inovation (pre-fit residual covariance)
    #K = np.linalg.solve(S, np.dot(P, H.T))
    K = np.dot(np.dot(P, H.T), np.linalg.inv(S))
    #print('S=', S, 'K=', K)
    #print('y - H*x=', y - np.dot(H, x))
    x = x + np.dot(K, y - np.dot(H, x))
    P = np.dot((np.eye(2) - np.dot(K, H)), P)

    t_est.append(t)
    x_est.append(x.tolist())
    P_est.append(P.tolist())

    # time update (skip time update at last epoch) 
    if _i >= len(t_idx)-1:
        break
    dt = t_idx[_i+1] - t
    t_prev = t
    F = np.array([\
        [1.0, dt],
        [0.0, 1.0]])
    Q = np.array([\
        [1.0 / 3.0 * np.power(sig0, 2) * np.power(dt, 3), 1.0 / 2.0 * np.power(sig0, 2) * np.power(dt, 2)], \
        [1.0 / 2.0 * np.power(sig0, 2) * np.power(dt, 2), np.power(sig0, 2)* dt]])
    x = np.dot(F, x)
    P = np.dot(np.dot(F, P), F.T) + Q
x_est = np.array(x_est)
P_est = np.array(P_est)

#
fig, axes = plt.subplots(2, 1)
plt.suptitle(f'Simulation (acc noise={sig0:.4f})')
pos_est = [v[0, 0] for v in x_est]
vel_est = [v[1, 0] for v in x_est]
axes[0].plot(t_est, pos_est,  label='KF fliter')
axes[0].plot(t_idx, [v[0] for v in x_true], label='true')
axes[0].plot(t_idx, pos_obs, '.', label="obs")
axes[0].set_ylabel('position [m]')
axes[1].plot(t_est, vel_est,  label='KF fliter')
axes[1].plot(t_idx, [v[1] for v in x_true], label='true')
axes[1].set_ylabel('velocity [m/s]')
for a in axes:
    a.grid(True)
    a.legend()

ofile = 'kf_1d_state_filter.png'
plt.savefig(ofile)
print(ofile)
# ---

t_smooth = []
x_smooth = []
P_smooth = []
n = len(t_idx)
x_smooth.append(x_est[n-1])
P_smooth.append(P_est[n-1])
t_smooth.append(t_idx[n-1])
_x_back = x_est[n-1]
_P_back = P_est[n - 1]
#_x_back = np.array([0.0, 0.0]).reshape(2, 1)
#_P_back = np.array([[100, 0.0], [0.0, 100.0]])
for _i in range(n-2, -1, -1):
    _x_pred = np.dot(F, x_est[_i])
    _P_pred = np.dot(np.dot(F, P_est[_i]), F.T) + Q
    #print('_i=', _i, ' P_{t|t}=', P_est[_i], ' P_{t+1|t}=', P_est[_i+1])
    J = np.dot(np.dot(P_est[_i], F.T), np.linalg.inv(_P_pred))
    #print('J=', J)
    #print('x_{t+1|T} - x_{t+1|t}=', _x_back - np.dot(F, x_est[_i]))
    _x_back = x_est[_i] + np.dot(J, (_x_back - _x_pred))
    _P_back = P_est[_i] + np.dot(J, np.dot(_P_back - _P_pred, J.T))
    t_smooth.append(t_idx[_i])
    x_smooth.append(_x_back.tolist())
    P_smooth.append(_P_back.tolist())
    #print('x_back=', _x_back, ' x_est=', x_est[_i], 'pos_true=', pos_true[_i])
    #print('P_back=', _P_back)

x_smooth = np.array(x_smooth)
P_smooth = np.array(P_smooth)

#
title_str = f'Simulation (acc noise={sig0:.4f})'
fig = plot_kf_1d_filter_smooth(t_est, x_est, t_smooth, x_smooth, t_idx, x_true, pos_obs, title_str)
ofile = 'kf_1d_state_smooth.png'
fig.savefig(ofile)
print(ofile)


#
# Plot vovariance matrix
ofile = 'kf_1d_state_var.png'
title_str = f'Simulation (acc noise={sig0:.4f})'
fig = plot_kf_1d_var(t_idx, P_est, P_smooth, title_str)
fig.savefig(ofile)
print(ofile)
