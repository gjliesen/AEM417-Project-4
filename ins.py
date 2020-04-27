# import pandas as pd
import numpy as np
import constants as cn
import attitude
import velocity
import position
import ekf
import update


def calc_rn(lat):
    return cn.a * (1 - cn.e ** 2) / ((1 - cn.e ** 2 * (np.sin(lat) ** 2)) ** 1.5)


def calc_re(lat):
    return cn.a / np.sqrt((1 - cn.e ** 2 * (np.sin(lat) ** 2)))


def lat_vector(lat):
    return np.array([[np.cos(lat)],
                     [0],
                     [-1 * np.sin(lat)]])


def calc_earth_rotation_matrix(lat):
    lat_vec = lat_vector(lat)
    return lat_vec * cn.omega_ei


def calc_trans_rate_matrix(rn, re, lat, h, vN):
    return np.array([[vN / (re + h)],
                     [(-1 * vN) / (rn + h)],
                     [(-1 * vN * np.tan(lat)) / (re + h)]])


def calc_c_bn_matrix(phi, theta, psi):
    c_bn = np.array(
        [[np.cos(theta) * np.cos(psi), np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi),
          np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)],
         [np.cos(theta) * np.sin(psi), np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi),
          np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)],
         [-1 * np.sin(theta), np.sin(phi) * np.cos(theta), np.cos(phi) * np.cos(theta)]])
    return c_bn


def get_values(ins_df, prev, psi_nb, p_e, v_n):
    rn = calc_rn(psi_nb[0][0])
    re = calc_re(psi_nb[0][0])
    wn_ie = calc_earth_rotation_matrix(p_e[0][0])
    wn_en = calc_trans_rate_matrix(rn, re, p_e[0][0], p_e[2][0], v_n[0][0])
    c_bn = calc_c_bn_matrix(psi_nb[0][0], psi_nb[1][0], psi_nb[2][0])
    g_n = velocity.g_n_matrix(p_e[0][0], p_e[1][0])
    wb_ib = ins_df.loc[prev, ['wX', 'wY', 'wZ']]
    wb_ib = wb_ib.to_numpy().reshape((-1, 1))
    return [rn, re, wn_ie, wn_en, c_bn, g_n, wb_ib]


def ins_formulation(ins_df, psi_nb, p_e, v_n, wb_ib, c_bn, wn_ie, wn_en, dt, rn, re, cur):
    att_cur = attitude.get(psi_nb[0][0], psi_nb[1][0], wb_ib, c_bn, psi_nb, wn_ie, wn_en, dt)
    ins_df.loc[cur, ['phi', 'theta', 'psi']] = att_cur.flatten()

    v_n_cur = velocity.get(p_e[2][0], p_e[0][0], wn_ie, wn_en, c_bn, wb_ib, v_n, dt)
    ins_df.loc[cur, ['vN', 'vE', 'vD']] = v_n_cur.flatten()

    pos_cur = position.get(rn, re, p_e[2][0], p_e[0][0], p_e, v_n, dt)
    ins_df.loc[cur, ['lat', 'long', 'h']] = pos_cur.flatten()

    return [ins_df, att_cur, v_n_cur, pos_cur]


def ekf_update(ins_df, del_x, re, rn, p_e, pos_gps, psi_nb, c_bn, v_n, cur):
    update.bias(del_x)

    v_n_update = update.vel(del_x, v_n)
    ins_df.loc[cur, ['vN', 'vE', 'vD']] = v_n_update.flatten()

    pos_update = update.pos(del_x, p_e, rn, re, pos_gps)
    ins_df.loc[cur, ['lat', 'long', 'h']] = pos_update.flatten()

    att_update = update.euler(psi_nb, c_bn)
    ins_df.loc[cur, ['phi', 'theta', 'psi']] = att_update.flatten()
    return [ins_df, att_update, v_n_update, pos_update]


def solve(ins_df, pos_df, vel_df):
    flag = True
    prev = np.nan
    for cur in ins_df.index:
        if flag:
            prev = cur
            flag = False
        else:
            print(cur)
            dt = ins_df.dt.loc[cur]

            # previous data
            psi_nb = attitude.extract_data(ins_df, prev)
            v_n = velocity.extract_data(ins_df, prev)
            p_e = position.extract_data(ins_df, prev)

            # Gather variables
            [rn, re, wn_ie, wn_en, c_bn, g_n, wb_ib] = get_values(ins_df, prev, psi_nb, p_e, v_n)

            # Predictions
            [ins_df, att_cur, v_n_cur, pos_cur] = ins_formulation(ins_df, psi_nb, p_e, v_n, wb_ib, c_bn, wn_ie,
                                                                  wn_en, dt, rn, re, cur)

            if cur % 1 == 0:
                # EKF Step
                pos_gps = position.extract_data(pos_df, cur)
                vel_gps = velocity.extract_data(vel_df, cur)
                del_x = ekf.get_state(pos_df, vel_df, cur, v_n_cur, pos_cur, c_bn, dt, wn_ie, wn_en, wb_ib, g_n,
                                      pos_gps, vel_gps)
                # Update Step
                ekf_update(ins_df, del_x, re, rn, p_e, pos_gps, psi_nb, c_bn, v_n, cur)
            # Iterate
            prev = cur
    print('Done')
    return ins_df
