import numpy as np
import matplotlib.pyplot as plt

Fs = 10
t_end = 10.0

sig0 = 1.0 # noise density of acceleration
sig1 = 0.5  # standard deviation of position observation

H = np.array([1, 0]).reshape(1, 2)
F = np.array([[1.0, 1.0 / Fs], [0.0, 1.0]])
R = [sig1 * sig1]
x_var_init = np.array([[10.0, 0.0], [0.0, 10.0]])
# Generate true and observed position
t_idx = np.arange(0.0, t_end, 1.0/Fs)
pos_true = 2.3 * (np.cos((4.0 * np.pi / t_end) * t_idx) + 1.2)
pos_obs = pos_true + np.random.normal(0.0, sig1*sig1, len(pos_true))

fig, axes = plt.subplots(2, 1)
axes[0].plot(t_idx, pos_true,  label='true')
axes[0].plot(t_idx, pos_obs, '.', label="obs")
axes[1].plot(t_idx[:-1], np.diff(pos_true), label='true')
for a in axes:
    a.grid(True)
    a.legend()

plt.savefig('posvel.png')

# Kalman Filter
x = np.array([0.0, 0.0]).reshape(2, 1)
P = x_var_init
t_est = []
x_est = []
P_est = []
t_prev = 0.0
for _i, (t, y) in enumerate(zip(t_idx, pos_obs)):
    t_est.append(t)
    x_est.append(x)
    P_est.append(P)

    if _i == 0:
        continue

    # time update 
    dt = t - t_prev
    t_prev = t
    Q = np.array([[1.0 / 3.0 * np.power(sig0, 2) * np.power(dt, 3), \
         1.0 / 2.0 * np.power(sig0, 2) * np.power(dt, 2)], \
        [1.0 / 2.0 * np.power(sig0, 2) * np.power(dt, 2), \
        np.power(sig0, 2)* dt]])
    x = np.dot(F, x)
    P = np.dot(np.dot(F, P), F.T) + Q
    #print(dt, y)
    #print('x=', x, 'P=', P)

    # measurement update
    S = np.dot(np.dot(H, P), H.T) + R  # inovation (pre-fit residual covariance)
    #K = np.linalg.solve(S, np.dot(P, H.T))
    K = np.dot(P, H.T) * (1.0 / S)
    #print('S=', S, 'K=', K)
    print('y - H*x=', y - np.dot(H, x))
    x = x + np.dot(K, y - np.dot(H, x))
    P = np.dot((np.eye(2) - np.dot(K, H)), P)

#
fig, axes = plt.subplots(2, 1)
plt.suptitle(f'Simulation (acc noise={sig0:.4f})')
pos_est = [v[0, 0] for v in x_est]
vel_est = [v[1, 0] for v in x_est]
axes[0].plot(t_est, pos_est,  label='KF fliter')
axes[0].plot(t_idx, pos_true, label='true')
axes[0].plot(t_idx, pos_obs, '.', label="obs")
axes[0].set_ylabel('position [m]')
axes[1].plot(t_est, vel_est,  label='KF fliter')
axes[1].plot(t_idx[:-1], np.diff(pos_true) / (1.0 / Fs), label='true')
axes[1].set_ylabel('velocity [m/s]')
for a in axes:
    a.grid(True)
    a.legend()

plt.savefig('kf_state_filter.png')
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
#_P_back = np.array([[100, 0.0], [0.0, 100.0]])
for _i in range(n-2, -1, -1):
    print(t_idx[_i], pos_obs[_i])
    _x_pred = np.dot(F, x_est[_i])
    _P_pred = np.dot(np.dot(F, P_est[_i]), F.T) + Q
    #print('_i=', _i, ' P_{t|t}=', P_est[_i], ' P_{t+1|t}=', P_est[_i+1])
    J = np.dot(np.dot(P_est[_i], F.T), np.linalg.inv(_P_pred))
    print('J=', J)
    print('x_{t+1|T} - x_{t+1|t}=', _x_back - np.dot(F, x_est[_i]))
    _x_back = x_est[_i] + np.dot(J, (_x_back - _x_pred))
    _P_back = P_est[_i] + np.dot(J, np.dot(_P_back - _P_pred, J.T))
    t_smooth.append(t_idx[_i])
    x_smooth.append(_x_back)
    P_smooth.append(_P_back)
    print('x_back=', _x_back, ' x_est=', x_est[_i], 'pos_true=', pos_true[_i])
    print('P_back=', _P_back)

#
fig, axes = plt.subplots(2, 1)
plt.suptitle(f'Simulation (acc noise={sig0:.4f})')
pos_est = [v[0, 0] for v in x_est]
pos_smooth = [v[0, 0] for v in x_smooth]
vel_est = [v[1, 0] for v in x_est]
vel_smooth = [v[1, 0] for v in x_smooth]
axes[0].plot(t_est, pos_est,  label='KF fliter')
axes[0].plot(t_smooth, pos_smooth,  label='KF smoothing')
axes[0].plot(t_idx, pos_true, label='true')
axes[0].plot(t_idx, pos_obs, '.', label="obs")
axes[0].set_ylabel('position [m]')
axes[1].plot(t_est, vel_est,  label='KF fliter')
axes[1].plot(t_smooth, vel_smooth,  label='KF smoothing')
axes[1].plot(t_idx[:-1], np.diff(pos_true) / (1.0 / Fs), label='true')
axes[1].set_ylabel('velocity [m/s]')
for a in axes:
    a.grid(True)
    a.legend()

plt.savefig('kf_state_smooth.png')

#
fig, axes = plt.subplots(2, 1)
plt.suptitle(f'Simulation (acc noise={sig0:.4f})')
pos_var_est = [v[0, 0] for v in P_est]
pos_var_smooth = [v[0, 0] for v in P_smooth]
vel_var_est = [v[1, 1] for v in P_est]
vel_var_smooth = [v[1, 1] for v in P_smooth]
axes[0].plot(t_est, pos_var_est,  label='KF fliter')
axes[0].plot(t_smooth, pos_var_smooth,  label='KF smoothing')
axes[0].set_ylabel(u'Var[$p(t)$] [m$^2$]')
axes[1].plot(t_est, vel_var_est,  label='KF fliter')
axes[1].plot(t_smooth, vel_var_smooth,  label='KF smoothing')
axes[1].set_ylabel(u'Var[$v(t)$] [m$^2$/s$^2$]')
for a in axes:
    a.grid(True)
    a.set_yscale('log')
    a.legend()

plt.savefig('kf_state_var.png')

