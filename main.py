import pandas as pd
import numpy as np
import math as m


def get_data(file):
    df = pd.read_csv(file, index_col=False, squeeze=True)
    return df


def get_imu_data(file):
    columns = ['Time','wX','wY','wZ','aX','aY','aZ']
    df = pd.read_csv(file, index_col=False, squeeze=True)
    df.columns = columns
    df = df.set_index('Time')
    return df


def init_ins_dataframe(index):
    columns = ['Lat', 'Long', 'h', 'phi', 'theta', 'psi', 'vX', 'vY', 'vZ']
    ins_df = pd.DataFrame(index = index, columns=columns)
    return ins_df


def calc_cntb_matrix(phi, theta, psi):
    data = [[np.cos(theta)*np.cos(psi), np.sin(phi)*np.sin(theta)*np.cos(psi) - np.cos(phi)*np.sin(psi),
             np.cos(phi)*np.sin(theta)*np.cos(psi) + np.sin(phi)*np.sin(psi)],
    [np.cos(theta) * np.sin(psi), np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi),
     np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)],
    [-1 * np.sin(theta), np.sin(phi) * np.cos(theta), np.cos(phi) * np.cos(theta)]]
    return pd.DataFrame(data)


def calc_a_matrix(phi, theta, psi):
    data = [[1, np.sin(phi)*np.sin(theta), np.cos(phi)*np.sin(theta)],
            [0, np.cos(phi)*np.cos(theta), -1 * np.sin(phi)*np.cos(theta)],
            [0, np.sin(phi), np.cos(phi)]]
    df = pd.DataFrame(data)
    df *= (1/np.cos(theta))
    return df


def main():
    # Parse data
    gps_lla_df = get_data('gps_pos_lla.txt')
    gps_vel_df = get_data('gps_vel_ned.txt')
    imu_df = get_imu_data('imu.txt')
    time_df = get_data('time.txt')
    ins_df = init_ins_dataframe(imu_df.index)





main()