import pandas as pd
import numpy as np


def get_vel_data(file):
    columns = ['vN', 'vE', 'vD']
    df = pd.read_csv(file, index_col=False, squeeze=True)
    df.columns = columns
    return df


def get_pos_data(file):
    columns = ['lat', 'long', 'h']
    df = pd.read_csv(file, index_col=False, squeeze=True)
    df.columns = columns
    return df


def get_time_data(file):
    df = pd.read_csv(file, index_col=False, squeeze=True)
    return df


def get_imu_data(file):
    columns = ['Time','wX','wY','wZ','aX','aY','aZ']
    df = pd.read_csv(file, index_col=False, squeeze=True)
    df.columns = columns
    return df


def initialize_data(imu_df, pos_df):
    columns=['Time', 'dt', 'lat', 'long', 'h', 'phi', 'theta', 'psi', 'vN', 'vE', 'vD']
    df = pd.DataFrame(index = imu_df.index, columns=columns)
    df.Time = imu_df.Time
    df.dt = df.Time - df.Time.shift(1)
    df.iloc[0,2] = pos_df.iloc[0,0]
    df.iloc[0,3] = pos_df.iloc[0,1]
    df.iloc[0,4] = pos_df.iloc[0,2]
    df.iloc[0,5:] = 0
    df = pd.merge(df, imu_df, on='Time')

    return df


def calc_rn(a, e, l):
    return a * (1 - e**2) / ((1 - e**2 * (np.sin(l)**2))**1.5)


def calc_re(a, e, l):
    return a / np.sqrt((1 - e**2 * (np.sin(l)**2)))


def get_skew(x):
    return pd.DataFrame(data=[[0, -1* x[2][0], x[1][0]],
                     [x[2][0], 0, -1 * x[0][0]],
                     [-1 * x[1][0], x[0][0], 0]])


def lat_vector(lat):
    return pd.DataFrame(data=[[np.cos(lat)], [0], [np.sin(lat)]])


def calc_earth_rotation_matrix(lat):
    # noinspection PyTypeChecker
    return lat_vector(lat) * 7.292115 * 10**(-5)


def calc_trans_rate_matrix(rn, re, lat, h, vN):
    return pd.DataFrame(data=[[vN / (re + h)], [(-1 * vN) / (rn + h)], [(-1 * vN * np.tan(lat))/(re + h)]])


def calc_c_bn_matrix(phi, theta, psi):
    data = [[np.cos(theta)*np.cos(psi), np.sin(phi)*np.sin(theta)*np.cos(psi) - np.cos(phi)*np.sin(psi),
             np.cos(phi)*np.sin(theta)*np.cos(psi) + np.sin(phi)*np.sin(psi)],
    [np.cos(theta) * np.sin(psi), np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi),
     np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)],
    [-1 * np.sin(theta), np.sin(phi) * np.cos(theta), np.cos(phi) * np.cos(theta)]]
    return pd.DataFrame(data)


def calc_a_nb_matrix(phi, theta):
    data = [[1, np.sin(phi) * np.sin(theta), np.cos(phi) * np.sin(theta)],
            [0, np.cos(phi) * np.cos(theta), -1 * np.sin(phi) * np.cos(theta)],
            [0, np.sin(phi), np.cos(phi)]]
    df = pd.DataFrame(data)
    df *= (1/np.cos(theta))
    return df


def attitude_update(ins_df, lat, rn, re, h, vN, phi, theta, psi, wb_ib, c_bn):
    wn_ie = calc_earth_rotation_matrix(lat)
    wn_en = calc_trans_rate_matrix(rn, re, lat, h, vN)
    wb_in = c_bn@(wn_ie + wn_en)
    a_nb = calc_a_nb_matrix(phi, theta)
    wb_nb = wb_ib - wb_in
    psiDot_nb = a_nb @ wb_nb


def calc_ch(a, omega, h, f, u_e):
    temp_1 = a**3 * (1 -f) * omega**2
    temp_2 = h / a
    return 1 - 2 * ( 1 + f + (temp_1 / u_e)) * temp_2 + 3 * temp_2**2


def calc_go(f, lat):
    return (9.7803253359 / np.sqrt(1 - f * (2 - f) * (np.sin(lat)**2))) * (1 + 0.0019311853*(np.sin(lat)**2))


def g_n_matrix(f, lat, a, omega_ei, h, u_e):
    go = calc_go(f, lat)
    ch = calc_ch(a, omega_ei, h, f, u_e)
    return pd.DataFrame(data=[[0],[0],[go * ch]])


def velocity_update(f, lat, a, omega_ei, h, u_e, wn_ie, wn_en, c_bn, f_b):
    g_n = g_n_matrix(f, lat, a, omega_ei, h, u_e)
    f_n = c_bn@f_b
    f_n + g_n - get_skew((2 * wn_ie - wn_en))@v_n


def pe_matrix(rn, re, h, lat):
    return pd.DataFrame(data=[[1 / (rn + h), 0, 0],
                     [0, 1/((re + h) * np.cos(lat)), 0],
                     [0, 0, -1]])


def position_update(rn, re, h, lat):
    pe_mat = pe_matrix(rn, re, h, lat)


def ins_formulation(imu_df, pos_df, vel_df, time_df):
    ins_df = initialize_data(imu_df, pos_df)
    a = 6378137
    f = 1 / 298.257223563
    e = np.sqrt(f * (2 - f))
    lat = ins_df.lat.iloc[0]
    u_e = 3.986004418 * 10**14
    omega_ei = 7.292115 * 10**(-5)
    rn = calc_rn(a, e, lat)
    re = calc_re(a, e, lat)
    phi = ins_df.iloc[0, 4]
    theta = ins_df.iloc[0, 5]
    psi = ins_df.iloc[0, 6]
    c_bn = calc_c_bn_matrix(phi, theta, psi)
    print('Done')



    print('Done')


def main():
    # Parse data
    pos_df = get_pos_data('gps_pos_lla.txt')
    vel_df = get_vel_data('gps_vel_ned.txt')
    imu_df = get_imu_data('imu.txt')
    time_df = get_time_data('time.txt')
    ins_formulation(imu_df, pos_df, vel_df, time_df)


main()