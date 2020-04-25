import pandas as pd
import numpy as np
import constants as cn


# import cProfile


def get_pos_data(file):
    columns = ['lat', 'long', 'h']
    df = pd.read_csv(file, index_col=False, squeeze=True)
    df.columns = columns
    return df


def get_vel_data(file):
    columns = ['vN', 'vE', 'vD']
    df = pd.read_csv(file, index_col=False, squeeze=True)
    df.columns = columns
    return df


def get_imu_data(file):
    columns = ['Time', 'wX', 'wY', 'wZ', 'aX', 'aY', 'aZ']
    df = pd.read_csv(file, index_col=False, squeeze=True)
    df.columns = columns
    return df


def get_time_data(file):
    df = pd.read_csv(file, index_col=False, squeeze=True)
    return df


def initialize_ins_data(imu_df, pos_df):
    columns = ['Time', 'dt', 'lat', 'long', 'h', 'phi', 'theta', 'psi', 'vN', 'vE', 'vD']
    ins_df = pd.DataFrame(index=imu_df.index, columns=columns)
    ins_df.Time = imu_df.Time
    ins_df.dt = ins_df.Time - ins_df.Time.shift(1)
    ins_df = pd.merge(ins_df, imu_df, on='Time')

    ins_df.set_index('Time', inplace=True)
    ins_df.lat.iat[0] = pos_df.lat.iat[0]
    ins_df.long.iat[0] = pos_df.long.iat[0]
    ins_df.h.iat[0] = pos_df.h.iat[0]
    ins_df.iloc[0, 4:10] = 0
    return ins_df


def calc_rn(lat):
    return cn.a * (1 - cn.e ** 2) / ((1 - cn.e ** 2 * (np.sin(lat) ** 2)) ** 1.5)


def calc_re(lat):
    return cn.a / np.sqrt((1 - cn.e ** 2 * (np.sin(lat) ** 2)))


def get_skew(x):
    return np.array([[0, -1 * x[2][0], x[1][0]],
                     [x[2][0], 0, -1 * x[0][0]],
                     [-1 * x[1][0], x[0][0], 0]])


def lat_vector(lat):
    return np.array([[np.cos(lat)],
                     [0],
                     [-1 * np.sin(lat)]])


def calc_earth_rotation_matrix(lat):
    # noinspection PyTypeChecker
    lat_vec = lat_vector(lat)
    return lat_vec * 7.292115 * 10 ** (-5)


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


def calc_a_nb_matrix(phi, theta):
    a_nb = np.array([[1, np.sin(phi) * np.sin(theta), np.cos(phi) * np.sin(theta)],
                     [0, np.cos(phi) * np.cos(theta), -1 * np.sin(phi) * np.cos(theta)],
                     [0, np.sin(phi), np.cos(phi)]])
    a_nb *= (1 / np.cos(theta))
    return a_nb


def attitude_update(phi, theta, wb_ib, c_bn, psi_nb, wn_ie, wn_en, dt):
    wb_in = c_bn @ (wn_ie + wn_en)
    a_nb = calc_a_nb_matrix(phi, theta)
    wb_nb = wb_ib - wb_in
    return (psi_nb + dt * a_nb @ wb_nb).T


def calc_ch(h):
    temp_1 = cn.a ** 3 * (1 - cn.f) * cn.omega_ei ** 2
    temp_2 = h / cn.a
    return 1 - 2 * (1 + cn.f + (temp_1 / cn.u_e)) * temp_2 + 3 * temp_2 ** 2


def calc_go(lat):
    return (9.7803253359 / np.sqrt(1 - cn.f * (2 - cn.f) * (np.sin(lat) ** 2))) * (1 + 0.0019311853 *
                                                                                   (np.sin(lat) ** 2))


def g_n_matrix(lat, h):
    go = calc_go(lat)
    ch = calc_ch(h)
    return np.array([[0],
                     [0],
                     [go * ch]])


def velocity_update(h, lat, wn_ie, wn_en, c_bn, wb_ib, v_n, dt):
    f_b = wb_ib + cn.acc_bias
    g_n = g_n_matrix(lat, h)
    f_n = c_bn @ f_b
    v_n_dot = f_n + g_n - get_skew(((2 * wn_ie) - wn_en)) @ v_n
    return v_n + dt * v_n_dot


def pe_matrix(rn, re, h, lat):
    return np.array([[1 / (rn + h), 0, 0],
                     [0, 1 / ((re + h) * np.cos(lat)), 0],
                     [0, 0, -1]])


def position_update(rn, re, h, lat, p_e, v_n, dt):
    pe_mat = pe_matrix(rn, re, h, lat)
    return p_e + dt * pe_mat @ v_n


def ins_formulation(ins_df):
    flag = True
    prev = np.nan
    for cur in ins_df.index:
        if flag:
            prev = cur
            flag = False
        else:
            # Constants
            lat = ins_df.lat.loc[prev]
            h = ins_df.h.loc[prev]
            phi = ins_df.phi.loc[prev]
            theta = ins_df.theta.loc[prev]
            psi = ins_df.theta.loc[prev]
            vN = ins_df.loc[prev, 'vN']
            dt = ins_df.dt.loc[cur]
            rn = calc_rn(lat)
            re = calc_re(lat)

            # Arrays
            psi_nb = ins_df.loc[prev, ['phi', 'theta', 'psi']]
            psi_nb = psi_nb.to_numpy().reshape((-1, 1))
            wb_ib = ins_df.loc[prev, ['aX', 'aY', 'aZ']]
            wb_ib = wb_ib.to_numpy().reshape((-1, 1))
            v_n = ins_df.loc[prev, ['vN', 'vE', 'vD']]
            v_n = v_n.to_numpy().reshape((-1, 1))
            p_e = ins_df.loc[prev, ['lat', 'long', 'h']]
            p_e = p_e.to_numpy().reshape((-1, 1))

            # Function Calls
            wn_ie = calc_earth_rotation_matrix(lat)
            wn_en = calc_trans_rate_matrix(rn, re, lat, h, vN)
            c_bn = calc_c_bn_matrix(phi, theta, psi)

            # Results
            ins_df.loc[cur, ['phi', 'theta', 'psi']] = attitude_update(phi, theta, wb_ib, c_bn, psi_nb, wn_ie,
                                                                       wn_en, dt).flatten()
            v_n_cur = velocity_update(h, lat, wn_ie, wn_en, c_bn, wb_ib, v_n, dt)
            ins_df.loc[cur, ['vN', 'vE', 'vD']] = v_n_cur.flatten()
            ins_df.loc[cur, ['lat', 'long', 'h']] = position_update(rn, re, h, lat, p_e, v_n, dt).flatten()

            # Iterate
            prev = cur
    print('Done')


def main():
    # Parse data

    pos_df = get_pos_data('gps_pos_lla.txt')
    # vel_df = get_vel_data('gps_vel_ned.txt')
    imu_df = get_imu_data('imu.txt')
    # time_df = get_time_data('time.txt')
    ins_df = initialize_ins_data(imu_df, pos_df)
    ins_formulation(ins_df)


main()
