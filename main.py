import pandas as pd
import numpy as np
import math as m



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
    df.set_index('Time', inplace=True)
    return df


def init_ins_dataframe(index):
    columns = ['lat', 'long', 'h', 'phi', 'theta', 'psi', 'vX', 'vY', 'vZ']
    df = pd.DataFrame(index = index, columns=columns)
    df.fillna(0)
    return df


def lat_vector(lat):
    return np.array([[np.cos(lat)], [0], [np.sin(lat)]])


def earth_rotation_matrix(lat):
    return 7.292115*10**(-5)*lat_vector(lat)

def calc_rn(a, e, l):
    return a * (1 - e**2) / ((1 - e**2 * (np.sin(l)**2))**1.5)


def calc_re(a, e, l):
    return a / np.sqrt((1 - e**2 * (np.sin(l)**2)))


def trans_rate_matrix(a, e, l, h, vN):
    a = 6378137
    rn = calc_rn(a, e, l)
    re = calc_re(a, e, l)
    return np.array([[vN / (re + h)], [(-1 * vN) / (rn + h)], [(-1 * vN * np.tan(l))/(re + h)]])

def calc_cntb_matrix(phi, theta, psi):
    data = [[np.cos(theta)*np.cos(psi), np.sin(phi)*np.sin(theta)*np.cos(psi) - np.cos(phi)*np.sin(psi),
             np.cos(phi)*np.sin(theta)*np.cos(psi) + np.sin(phi)*np.sin(psi)],
    [np.cos(theta) * np.sin(psi), np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi),
     np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)],
    [-1 * np.sin(theta), np.sin(phi) * np.cos(theta), np.cos(phi) * np.cos(theta)]]
    return pd.DataFrame(data)


def calc_a_matrix(phi, theta):
    data = [[1, np.sin(phi)*np.sin(theta), np.cos(phi)*np.sin(theta)],
            [0, np.cos(phi)*np.cos(theta), -1 * np.sin(phi)*np.cos(theta)],
            [0, np.sin(phi), np.cos(phi)]]
    df = pd.DataFrame(data)
    df *= (1/np.cos(theta))
    return df

def ins_formulation(imu_df, pos_df, vel_df, time_df):
    ins_df = init_ins_dataframe(imu_df.index)
    ins_df.iloc[0,0] = pos_df.iloc[0,0]
    ins_df.iloc[0,1] = pos_df.iloc[0,1]
    ins_df.iloc[0,2] = pos_df.iloc[0,2]
    ins_df['E_rot'] = ins_df.lat.transform(lambda x: earth_rotation_matrix(x))
    print('Done')

def main():
    # Parse data
    pos_df = get_pos_data('gps_pos_lla.txt')
    vel_df = get_vel_data('gps_vel_ned.txt')
    imu_df = get_imu_data('imu.txt')
    time_df = get_time_data('time.txt')
    ins_formulation(imu_df, pos_df, vel_df, time_df)


main()