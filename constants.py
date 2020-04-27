import numpy as np


def get_skew(x):
    return np.array([[0, -1 * x[2][0], x[1][0]],
                     [x[2][0], 0, -1 * x[0][0]],
                     [-1 * x[1][0], x[0][0], 0]])


g = 9.8
a: float = 6378137
f = 1 / 298.257223563
e = np.sqrt(f * (2 - f))
u_e = 3.986004418 * 10 ** 14
omega_ei = 7.292115 * 10 ** (-5)
sigma_p = 3
sigma_v = 0.2
acc_bias = np.array([[0.25],
                     [0.055],
                     [-0.12]])
sigma_b_acc = 0.0005 * g
tau_acc = 300
sigma_w_acc = 0.12 * g
gyro_bias = np.array([[2.4*10**-4],
                      [-1.3*10**-4],
                      [5.6*10**-4]])
sigma_b_gyro = np.deg2rad(0.3)
tau_gyro = 300
sigma_w_gyro = np.deg2rad(0.95)

sigma_u_acc = np.sqrt((2 * sigma_b_acc**2)/tau_acc)
sigma_u_gyro = np.sqrt((2 * sigma_b_gyro**2)/tau_gyro)

S = np.diag([sigma_w_acc ** 2, sigma_w_acc ** 2, sigma_w_acc ** 2, sigma_w_gyro ** 2,
             sigma_w_gyro ** 2, sigma_w_gyro ** 2, sigma_u_acc, sigma_u_acc, sigma_u_acc,
             sigma_u_gyro, sigma_u_gyro, sigma_u_gyro])

R = np.diag([sigma_p, sigma_p, sigma_p, sigma_v, sigma_v, sigma_v])


H = np.array([[np.identity(6), np.zeros((6, 9))]])
