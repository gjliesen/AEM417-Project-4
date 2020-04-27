import numpy as np
import constants as cn
import math as m


def bias(del_x, gyro_bias, acc_bias):
    acc_bias = acc_bias - del_x[9:12]
    gyro_bias = gyro_bias - del_x[12:15]
    return [acc_bias, gyro_bias]


def pos(del_x, p_e, rn, re, pos_gps):
    lat = p_e[0][0] - (del_x[0][0] / (rn + p_e[2][0]))
    long = p_e[1][0] - del_x[1][0] / (re + p_e[2][0] * np.cos(p_e[1][0]))
    h = pos_gps[2][0]
    return np.array([[lat],
                     [long],
                     [h]])


def vel(del_x, v_n_cur, vel_gps):
    vN = v_n_cur[0][0] - del_x[3][0]
    vE = v_n_cur[1][0] - del_x[4][0]
    vD = vel_gps[2][0]
    v_n_update = np.array([[vN],
                           [vE],
                           [vD]])
    return v_n_update


def c_bn_matrix(psi_error, c_nb):
    return (np.identity(3) + cn.get_skew(psi_error)) @ c_nb


def euler(c_bn, del_x):
    psi_error = del_x[5:8]
    c_bn = c_bn_matrix(psi_error, c_bn)
    phi = m.atan2(c_bn[2][1], c_bn[2][2])
    theta = -1 * m.asin(c_bn[2][0])
    psi = m.atan2(c_bn[1][0], c_bn[0][0])
    return np.array([[phi],
                     [theta],
                     [psi]])
