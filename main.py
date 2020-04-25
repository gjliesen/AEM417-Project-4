import pandas as pd
import numpy as np
import cProfile


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


def calc_rn(a, e, lat):
    return a * (1 - e ** 2) / ((1 - e ** 2 * (np.sin(lat) ** 2)) ** 1.5)


def calc_re(a, e, lat):
    return a / np.sqrt((1 - e ** 2 * (np.sin(lat) ** 2)))


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


def attitude_update(phi, theta, wb_ib, c_bn, psi_nb, dt, wn_ie, wn_en):
    wb_in = c_bn @ (wn_ie + wn_en)
    a_nb = calc_a_nb_matrix(phi, theta)
    wb_nb = wb_ib - wb_in
    return (psi_nb + dt * a_nb @ wb_nb).T


def calc_ch(a, omega, h, f, u_e):
    temp_1 = a ** 3 * (1 - f) * omega ** 2
    temp_2 = h / a
    return 1 - 2 * (1 + f + (temp_1 / u_e)) * temp_2 + 3 * temp_2 ** 2


def calc_go(f, lat):
    return (9.7803253359 / np.sqrt(1 - f * (2 - f) * (np.sin(lat) ** 2))) * (1 + 0.0019311853 * (np.sin(lat) ** 2))


def g_n_matrix(f, lat, a, omega_ei, h, u_e):
    go = calc_go(f, lat)
    ch = calc_ch(a, omega_ei, h, f, u_e)
    return np.array([[0],
                     [0],
                     [go * ch]])


def velocity_update(f, lat, a, omega_ei, h, u_e, wn_ie, wn_en, c_bn, wb_ib, v_n, dt):
    acc_bias = np.array([[0.25],
                         [0.055],
                         [-0.12]])
    f_b = wb_ib + acc_bias
    g_n = g_n_matrix(f, lat, a, omega_ei, h, u_e)
    f_n = c_bn @ f_b
    v_n_dot = f_n + g_n - get_skew(((2 * wn_ie) - wn_en)) @ v_n
    return v_n + dt * v_n_dot


def pe_matrix(rn, re, h, lat):
    return np.array([[1 / (rn + h), 0, 0],
                     [0, 1 / ((re + h) * np.cos(lat)), 0],
                     [0, 0, -1]])


def position_update(rn, re, h, lat):
    pe_mat = pe_matrix(rn, re, h, lat)


def ins_formulation(ins_df, vel_df, time_df):
    a = 6378137
    f = 1 / 298.257223563
    e = np.sqrt(f * (2 - f))
    u_e = 3.986004418 * 10 ** 14
    omega_ei = 7.292115 * 10 ** (-5)

    flag = True
    prev = np.nan
    for cur in ins_df.index:
        if flag:
            prev = cur
            flag = False
        else:
            # Constants
            lat = ins_df.lat.loc[prev]
            long = ins_df.long.loc[prev]
            h = ins_df.h.loc[prev]
            phi = ins_df.phi.loc[prev]
            theta = ins_df.theta.loc[prev]
            psi = ins_df.theta.loc[prev]
            vN = ins_df.loc[prev, 'vN']
            dt = ins_df.dt.loc[cur]
            rn = calc_rn(a, e, lat)
            re = calc_re(a, e, lat)
            # Arrays

            psi_nb = ins_df.loc[prev, ['phi', 'theta', 'psi']]
            psi_nb = psi_nb.to_numpy().reshape((-1, 1))
            wb_ib = ins_df.loc[prev, ['aX', 'aY', 'aZ']]
            wb_ib = wb_ib.to_numpy().reshape((-1, 1))
            v_n = ins_df.loc[prev, ['vN', 'vE', 'vD']]
            v_n = v_n.to_numpy().reshape((-1, 1))
            wn_ie = calc_earth_rotation_matrix(lat)
            wn_en = calc_trans_rate_matrix(rn, re, lat, h, vN)
            #
            # print(wb_ib)
            c_bn = calc_c_bn_matrix(phi, theta, psi)
            ins_df.loc[cur, ['phi', 'theta', 'psi']] = attitude_update(phi, theta, wb_ib, c_bn, psi_nb, dt).flatten()
            ins_df.loc[cur, ['vN', 'vE', 'vD']] = velocity_update(f, lat, a, omega_ei, h, u_e,
                                                                  wn_ie, wn_en, c_bn, wb_ib, v_n, dt).flatten()
            prev = cur
    print('Done')


def main():
    # Parse data

    pos_df = get_pos_data('gps_pos_lla.txt')
    vel_df = get_vel_data('gps_vel_ned.txt')
    imu_df = get_imu_data('imu.txt')
    time_df = get_time_data('time.txt')
    ins_df = initialize_ins_data(imu_df, pos_df)
    ins_formulation(ins_df, vel_df, time_df)


main()
