import numpy as np

def generate_true_pos_vel_acc_type1(t_idx):
    """
    """
    A, W = 5.0, 2.0 * np.pi / 10.0
    pos_true = A * (np.cos(W * t_idx) - 1.0)
    vel_true = A * (-W * np.sin(W * t_idx))
    acc_true = A * (-W * W * np.sin(W * t_idx))
    #pos_true, vel_true, acc_true = np.zeros(len(t_idx)), np.zeros(len(t_idx)), np.zeros(len(t_idx))
    x_true = np.array([[_x, 0, 0, _v, 0, 0, _a, 0, 0] for _x, _v, _a in zip(pos_true, vel_true, acc_true)])
    return x_true


def generte_true_pos_vel_1d_type1(t_idx, motion_period_time, pos_offset:float = 1.0):
    """
    """
    W = 2.0 * np.pi / motion_period_time
    pos_true = 2.3 * (np.cos(W * t_idx)  - pos_offset)
    vel_true = 2.3 * W * (-np.sin(W * t_idx))
    x_true = np.array([ [_x, _v] for _x, _v in zip(pos_true, vel_true)])
    return x_true

def kf_pv_1d_type1_condition():
    """
    Return a condition.
    - noise density of acceleration
    - variance of positoin observation noise
    - 
    """
    fs = 2
    t_end = 60.0
    x0 = np.array([0.0, 0.0])
    x0_var = np.array([[4.0, 0.0], [0.0, 4.0]])
    sig_acc, sig_pos = 1.0, 0.5
    return fs, t_end, sig_acc, sig_pos, x0, x0_var
