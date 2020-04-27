import numpy as np
import constants as cn
from scipy.linalg import expm
from navpy import lla2ned
import position


def get_fb(wb_ib):
    f_b = wb_ib + cn.acc_bias
    return f_b


def calc_wn_in(wn_ie, wn_en):
    return wn_ie + wn_en


def calc_wb_nb(wn_in, wb_ib):
    return wb_ib - wn_in + cn.gyro_bias


def state_matrix_a(wn_en, g_n, wn_ie, c_bn, f_b, wn_in):
    row_1 = np.hstack((-1 * cn.get_skew(wn_en), np.identity(3), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))))
    row_2 = np.hstack(((np.abs(g_n) / cn.a) * np.diag([-1, 1, 2]), -1 * cn.get_skew(2 * wn_ie + wn_en),
                       cn.get_skew(c_bn @ f_b), c_bn, np.zeros((3, 3))))
    row_3 = np.hstack((np.zeros((3, 3)), np.zeros((3, 3)), -1 * cn.get_skew(wn_in), np.zeros((3, 3)), -1 * c_bn))
    row_4 = np.hstack((np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), (-1 / cn.tau_acc) * np.identity(3),
                       np.zeros((3, 3))))
    row_5 = np.hstack((np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), (-1 / cn.tau_gyro) *
                       np.identity(3)))
    A = np.vstack((row_1, row_2, row_3, row_4, row_5))
    return A


def noise_model_matrix_m(c_bn):
    row_1 = np.hstack((np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))))
    row_2 = np.hstack((c_bn, np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))))
    row_3 = np.hstack((np.zeros((3, 3)), -1 * c_bn, np.zeros((3, 3)), np.zeros((3, 3))))
    row_4 = np.hstack((np.zeros((3, 3)), np.zeros((3, 3)), np.identity(3), np.zeros((3, 3))))
    row_5 = np.hstack((np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.identity(3)))
    M = np.vstack((row_1, row_2, row_3, row_4, row_5))
    return M


def get_f_matrix(A, dt):
    F = expm(A * dt)
    return F


def noise_covariance_q(A, M, dt):
    Q = (np.identity(15) + dt * A) @ (dt * M @ cn.S @ M.T)
    return Q


def calc_position_p_kk(pos_cur, F, Q):
    p_kk = F @ cn.p_init @ F.T + Q
    p_kk = 0.5 * (p_kk + p_kk.T)
    return p_kk


def noise_gain_matrix_k(p_kk):
    K = p_kk @ cn.H.T @ np.linalg.inv(cn.H @ p_kk @ cn.H.T + cn.R)
    return K


def loose_state_matrix(v_n_cur, pos_cur, K, pos_gps, vel_gps):
    temp = np.vstack((pos_gps - pos_cur,
                     vel_gps - v_n_cur))
    del_x = K @ temp
    return del_x


def gnss_vs_predicted(del_x, v_n_cur):
    del_y = cn.H @ del_x + v_n_cur
    return del_y


def get_state(pos_df, vel_df, cur, v_n_cur, pos_cur, c_bn, dt, wn_ie, wn_en, wb_ib, g_n, pos_gps, vel_gps):
    f_b = get_fb(wb_ib)
    wn_in = calc_wn_in(wn_ie, wn_en)
    wb_nb = calc_wb_nb(wn_in, wb_ib)
    pos_ref = position.extract_data(pos_df, pos_df.index[0])
    pos_cur_ned = lla2ned(pos_cur[0][0], pos_cur[1][0], pos_cur[2][0], pos_ref[0][0], pos_ref[1][0], pos_ref[2][0])
    pos_gps_ned = lla2ned(pos_gps[0][0], pos_gps[1][0], pos_gps[2][0], pos_ref[0][0], pos_ref[1][0], pos_ref[2][0])
    pos_cur_ned = pos_cur_ned.reshape((-1, 1))
    pos_gps_ned = pos_gps_ned.reshape((-1, 1))
    A = state_matrix_a(wn_en, g_n, wn_ie, c_bn, f_b, wn_in)
    M = noise_model_matrix_m(c_bn)
    F = get_f_matrix(A, dt)
    Q = noise_covariance_q(A, M, dt)
    p_kk = calc_position_p_kk(pos_cur_ned, F, Q)
    K = noise_gain_matrix_k(p_kk)
    del_x = loose_state_matrix(v_n_cur, pos_cur_ned, K, pos_gps_ned, vel_gps)
    return del_x
