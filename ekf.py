import pandas as pd
import numpy as np
import constants as cn
import attitude
import velocity
import position
from scipy.linalg import expm


def state_matrix_a(wn_en, g_n, wn_ie, c_nb, f_b, wn_in):
    row_1 = np.array([-1 * cn.get_skew(wn_en), np.identity(3), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))])
    row_2 = np.array([(np.abs(g_n)/cn.a) * np.diagonal(-1, 1, 2), -1 * cn.get_skew(2 * wn_ie + wn_en),
                      cn.get_skew(c_nb @ f_b), c_nb, np.zeros((3, 3))])
    row_3 = np.array([np.zeros((3, 3)), np.zeros((3, 3)), -1 * cn.get_skew(wn_in), np.zeros((3, 3)), -1 * c_nb])
    row_4 = np.array([np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), (-1 / cn.tau_acc) * np.identity(3),
                      np.zeros((3, 3))])
    row_5 = np.array([np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), (-1 / cn.tau_gyro) *
                      np.identity(3)])
    A = np.vstack((row_1, row_2, row_3, row_4, row_5))
    return A


def noise_model_matrix_m(c_nb):
    row_1 = np.array([np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))])
    row_2 = np.array([c_nb, np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))])
    row_3 = np.array([np.zeros((3, 3)), -1 * c_nb, np.zeros((3, 3)), np.zeros((3, 3))])
    row_4 = np.array([np.zeros((3, 3)), np.zeros((3, 3)), np.identity(3), np.zeros((3, 3))])
    row_5 = np.array([np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.identity(3)])
    M = np.vstack((row_1, row_2, row_3, row_4, row_5))
    return M


def noise_covariance_q(A, M, dt):
    F = expm(A * dt)
    Q = (np.identity(15) + dt * A) @ (dt * M @ S @ M.T)


def calc_wn_in(wn_ie, wn_en):
    return wn_ie + wn_en


def calc_wb_nb(wn_in, wb_ib):
    return wb_ib - wn_in + cn.gyro_bias


def noise_gain_matrix():
    print('Placeholder')


def gnss_noise_matrix():
    H = np.array([[np.identity(6), np.zeros((6, 9))]])
    return


def r_diagonal_matrix():
    R = np.diag([cn.sigma_p, cn.sigma_p, cn.sigma_p, cn.sigma_v, cn.sigma_v, cn.sigma_v])
    return R


def loose_state_matrix(pos_df, vel_df, ins_df, cur, att_cur, v_n_cur, pos_cur, psi_nb):
    pos_gps = position.extract_data(pos_df, cur)
    vel_gps = velocity.extract_data(vel_df, cur)
    del_x = np.array([[pos_gps - pos_cur],
                      [vel_gps - v_n_cur],
                      [psi_nb]])
    print('placeholder')
