import numpy as np
import constants as cn
import attitude
import velocity
import position


def bias(del_x):
    cn.acc_bias = cn.acc_bias - del_x[9:12]
    cn.gyro_bias = cn.gyro_bias - del_x[12:15]


def pos(del_x, p_e, rn, re, pos_gps):
    lat = p_e[0][0] - (del_x[0][0] / (rn + p_e[2][0]))
    long = p_e[1][0] - (del_x[1][0] / ((re + p_e[2][0]) * np.cos(p_e[1][0])))
    h = pos_gps[2][0] + del_x[2][0]
    return np.array([[lat],
                     [long],
                     [h]])


def vel(del_x, v_n_cur):
    return v_n_cur - del_x[2:5]


def c_bn_matrix(psi_nb, c_nb):
    return (np.identity(3) + cn.get_skew(psi_nb)) @ c_nb


def euler(psi_nb, c_bn):
    c_bn = c_bn_matrix(psi_nb, c_bn)
    phi = np.arctan(c_bn[2][1] / c_bn[2][2])
    theta = -1 * np.arcsin(c_bn[2][0])
    psi = np.arctan(c_bn[1][0] / c_bn[0][0])
    return np.array([[phi],
                     [theta],
                     [psi]])
