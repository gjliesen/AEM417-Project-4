# import pandas as pd
import numpy as np
import constants as cn
import attitude
import velocity
import position


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


def get_values(psi_nb, p_e, v_n):
    rn = calc_rn(psi_nb[0][0])
    re = calc_re(psi_nb[0][0])
    wn_ie = calc_earth_rotation_matrix(p_e[0][0])
    wn_en = calc_trans_rate_matrix(rn, re, p_e[0][0], p_e[0][2], v_n[0][0])
    c_bn = calc_c_bn_matrix(psi_nb[0][0], psi_nb[0][1], psi_nb[0][2])
    return [rn, re, wn_ie, wn_en, c_bn]


def update_predictions(ins_df, psi_nb, p_e, v_n, wb_ib, c_bn, wn_ie, wn_en, dt, rn, re, cur):
    att_cur = attitude.update(psi_nb[0][0], psi_nb[0][1], wb_ib, c_bn, psi_nb, wn_ie, wn_en, dt)
    ins_df.loc[cur, ['phi', 'theta', 'psi']] = att_cur.flatten()
    v_n_cur = velocity.update(p_e[0][2], p_e[0][0], wn_ie, wn_en, c_bn, wb_ib, v_n, dt)
    ins_df.loc[cur, ['vN', 'vE', 'vD']] = v_n_cur.flatten()
    pos_cur = position.update(rn, re, p_e[0][2], p_e[0][0], p_e, v_n, dt)
    ins_df.loc[cur, ['lat', 'long', 'h']] = pos_cur.flatten()
    return [ins_df, att_cur, v_n_cur, pos_cur]


def ins_formulation(ins_df):
    flag = True
    prev = np.nan
    for cur in ins_df.index:
        if flag:
            prev = cur
            flag = False
        else:
            dt = ins_df.dt.loc[cur]

            # previous data
            psi_nb = attitude.extract_data(ins_df, prev)
            v_n = velocity.extract_data(ins_df, prev)
            p_e = position.extract_data(ins_df, prev)

            # build accelerometer array
            wb_ib = ins_df.loc[prev, ['aX', 'aY', 'aZ']]
            wb_ib = wb_ib.to_numpy().reshape((-1, 1))

            # Gather variables
            [rn, re, wn_ie, wn_en, c_bn] = get_values(psi_nb, v_n, p_e)

            # Predictions
            [ins_df, att_cur, v_n_cur, pos_cur] = update_predictions(ins_df, psi_nb, p_e, v_n, wb_ib, c_bn, wn_ie,
                                                                     wn_en, dt, rn, re, cur)
            # Iterate
            prev = cur
    print('Done')
    return ins_df
