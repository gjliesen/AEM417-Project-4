import numpy as np
g = 9.8
a: float = 6378137
f = 1 / 298.257223563
e = np.sqrt(f * (2 - f))
u_e = 3.986004418 * 10 ** 14
omega_ei = 7.292115 * 10 ** (-5)
pos_std = 3
vel_std = 0.2
acc_bias = np.array([[0.25],
                     [0.055],
                     [-0.12]])
acc_bias_std = 0.0005 * g
tau_acc = 300
acc_noise_std = 0.12 * g
gyro_bias = np.array([[2.4*10**-4],
                      [-1.3*10**-4],
                      [5.6*10**-4]])
gyro_bias_std = 0.3
tau_gyro = 300
gyro_noise_std = 0.95
