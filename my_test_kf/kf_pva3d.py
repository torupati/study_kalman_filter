import numpy as np
import matplotlib.pyplot as plt

Fs = 10
t_end = 10.0

sig_acc = np.diag([0.0, 0.0, 0.0]) # noise density of acceleration
R = [[0.6, 0.0, 0.0],
    [0.0, 0.5, -0.0], \
    [0.0, -0.0, 0.2]] # covariance of position observation

H = np.array([\
    [1, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 1, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 1, 0, 0, 0, 0, 0, 0]]).reshape(3, 9)

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
A, W = 5.0, 3.0 * np.pi / t_end
#pos_true = A * (np.cos(W * t_idx) + 1.2)
#vel_true = A * (-W * np.sin(W * t_idx))
#acc_true = A * (-W * W * np.sin(W * t_idx))
pos_true, vel_true, acc_true = np.zeros(len(t_idx)), np.zeros(len(t_idx)), np.zeros(len(t_idx))
x_true = np.array([[_x, 0, 0, _v, 0, 0, _a, 0, 0] for _x, _v, _a in zip(pos_true, vel_true, acc_true)])
pos_obs = np.array([[_x, 0, 0] + np.random.multivariate_normal(np.zeros(3), R) for _x in pos_true])

# ---
#fig, axes = plt.subplots(2, 1, figsize=(10,6))
#for _d in range(3):
#    axes[0].plot(t_idx, x_true[:, _d],  label='true')
#    axes[0].plot(t_idx, pos_obs[:, _d], '.', label="obs")
#axes[1].plot(t_idx[:-1], np.diff(pos_true), label='true')
#for a in axes:
#    a.grid(True)
#    a.legend()
#plt.savefig('posvel3d.png')

# ---
# Kalman Filter
x = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(9, 1)
P = x_var_init
t_est = []
x_est = []
P_est = []
t_pred, x_pred, P_pred = [], [], []
t_prev = 0.0
for i, (t, y) in enumerate(zip(t_idx, pos_obs)):
    t_est.append(t)
    x_est.append(x)
    P_est.append(P)

    if i == 0:
        t_prev = t
        continue

    # time update 
    dt = t - t_prev
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
    Q[0:3, 6:9] = Q[6:9, 0:3] =  np.power(dt, 3) * sig_acc #* (1.0 / 6.0)# E[p(t)a(t)]
    Q[3:6, 3:6] = 1.0 / 3.0 * np.power(dt, 3) * sig_acc # E[v(t)v(t)]
    Q[3:6, 0:3] = Q[3:6, 0:3] = 1.0 / 2.0 * np.power(dt, 2) * sig_acc # E[v(t)a(t)]
    Q[6:9, 6:9] = dt * sig_acc # E[a(t)a(t)]
    x = np.dot(F, x)
    P = np.dot(np.dot(F, P), F.T) + Q
    #print(dt, y.shape, 'y=', y)
    #print('x=', x, 'P=', P)
    t_pred.append(t)
    x_pred.append(x)
    P_pred.append(P)

    # measurement update
    S = np.dot(np.dot(H, P), H.T) + R  # inovation (pre-fit residual covariance)
    #K = np.linalg.solve(S, np.dot(P, H.T))
    K = np.dot(np.dot(P, H.T), np.linalg.inv(S))
    #print('S=', S, 'K=', K)
    #print('y - H*x=', y.reshape(3,1) - np.dot(H, x))
    x = x + np.dot(K, y.reshape(3, 1) - np.dot(H, x))
    P = np.dot((np.eye(9) - np.dot(K, H)), P)
    t_prev = t

# ------
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
plt.suptitle(f'Simulation')
vel_est = [v[1, 0] for v in x_est]
for _d, _lb in enumerate(['X', 'Y', 'Z']):
    axes[_d,0].plot(t_idx, [v[_d] for v in x_true], label='true')
    axes[_d,0].plot(t_est, [v[_d] for v in x_est], '.', label=f'KF fliter {_lb}')
    axes[_d,0].plot(t_pred, [v[_d] for v in x_pred], '.', label=f'prediction {_lb}')
    axes[_d,0].plot(t_idx, [v[_d] for v in pos_obs], 'o', label="obs")
    axes[_d,0].set_ylabel(f'position {_lb} [m]')
for _d, _lb in enumerate(['X', 'Y', 'Z']):
    axes[_d,1].plot(t_idx, [v[3+_d] for v in x_true], label=f'true {_lb}')
    axes[_d,1].plot(t_est, [v[3+_d] for v in x_est], '.', label=f'KF fliter {_lb}')
    axes[_d,1].plot(t_pred, [v[3+_d] for v in x_pred], '.',  label=f'prediction {_lb}')
    axes[_d,1].set_ylabel(f'Velocity {_lb} [m]')
for _d, _lb in enumerate(['X', 'Y', 'Z']):
    axes[_d,2].plot(t_idx, [v[6+_d] for v in x_true], label=f'true')
    axes[_d,2].plot(t_est, [v[6+_d] for v in x_est], '.', label='KF fliter')
    axes[_d,2].plot(t_pred, [v[6+_d] for v in x_pred], '.', label='prediction')
    axes[_d,2].set_ylabel(u'acc [m/s$^2$]')
for i in range(axes.shape[0]):
    for j in range(axes.shape[1]):
        axes[i,j].grid(True)
        axes[i,j].legend(loc='upper left')

ofile = 'kf_3d_state_filter.png'
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
_P_back = P_est[n-1]
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
    x_smooth.append(_x_back)
    P_smooth.append(_P_back)
    #print('x_back=', _x_back, ' x_est=', x_est[_i], 'pos_true=', pos_true[_i])
    #print('P_back=', _P_back)

# ------
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
plt.suptitle(f'Simulation')
vel_est = [v[1, 0] for v in x_est]
for _d, _lb in enumerate(['X', 'Y', 'Z']):
    axes[_d,0].plot(t_est, [v[_d] for v in x_est],  label=f'KF fliter')
    axes[_d,0].plot(t_est, [v[_d] for v in x_smooth],  label=f'KF smoother')
    axes[_d,0].plot(t_idx, [v[_d] for v in x_true], label='true')
    axes[_d,0].plot(t_idx, [v[_d] for v in pos_obs], 'o', label="obs")
    axes[_d,0].set_ylabel(f'position {_lb} [m]')
for _d, _lb in enumerate(['X', 'Y', 'Z']):
    axes[_d,1].plot(t_est, [v[3+_d] for v in x_est],  label=f'KF fliter')
    axes[_d,1].plot(t_est, [v[3+_d] for v in x_smooth],  label=f'KF smoother')
    axes[_d,1].plot(t_idx, [v[3+_d] for v in x_true], label=f'true {_lb}')
    axes[_d,1].set_ylabel(f'Velocity {_lb} [m]')
for _d, _lb in enumerate(['X', 'Y', 'Z']):
    axes[_d,2].plot(t_est, [v[6+_d] for v in x_est],  label=f'KF fliter')
    axes[_d,2].plot(t_est, [v[6+_d] for v in x_smooth],  label=f'KF smoother')
    axes[_d,2].plot(t_idx, [v[6+_d] for v in x_true], label=f'true {_lb}')
    axes[_d,2].set_ylabel(u'acc [m/s$^2$]')
for i in range(axes.shape[0]):
    for j in range(axes.shape[1]):
        axes[i,j].grid(True)
        axes[i,j].legend(loc='upper left')

ofile = 'kf_3d_state_smoother.png'
plt.savefig(ofile)
print(ofile)


# ------
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
plt.suptitle(f'Simulation')
for _d, _lb in enumerate(['X', 'Y', 'Z']):
    axes[_d,0].plot(t_est, [v[_d, _d] for v in P_est],  label='KF fliter')
    axes[_d,0].plot(t_smooth, [v[_d, _d] for v in P_smooth],  label='KF smoothing')
for _d, _lb in enumerate(['X', 'Y', 'Z']):
    axes[_d,1].plot(t_est, [v[3+_d, 3+_d] for v in P_est],  label='KF fliter')
    axes[_d,1].plot(t_smooth, [v[3+_d, 3+_d] for v in P_smooth],  label='KF smoothing')
for _d, _lb in enumerate(['X', 'Y', 'Z']):
    axes[_d,2].plot(t_est, [v[6+_d, 6+_d] for v in P_est],  label='KF fliter')
    axes[_d,2].plot(t_smooth, [v[6+_d, 6+_d] for v in P_smooth],  label='KF smoothing')
for i in range(axes.shape[0]):
    for j in range(axes.shape[1]):
        axes[i,j].grid(True)
        axes[i,j].legend(loc='upper left')
        axes[i,j].set_yscale('log')

#axes[0].set_ylabel(u'Var[$p(t)$] [m$^2$]')
#axes[1].plot(t_est, vel_var_est,  label='KF fliter')
#axes[1].plot(t_smooth, vel_var_smooth,  label='KF smoothing')
#axes[1].set_ylabel(u'Var[$v(t)$] [m$^2$/s$^2$]')

ofile = 'kf_3d_state_var.png'
plt.savefig(ofile)
print(ofile)
