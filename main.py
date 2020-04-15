import pandas as pd
import numpy as np
import math as m

def get_data(file):
    df = pd.read_csv(file, index_col=False, squeeze=True)
    return df

def calc_c_n_to_b(phi, theta, psi):
    data = [[np.cos(theta)*np.cos(psi), np.sin(phi)*np.sin(theta)*np.cos(psi) - np.cos(phi)*np.sin(psi),
             np.cos(phi)*np.sin(theta)*np.cos(psi) + np.sin(phi)*np.sin(psi)],
    [np.cos(theta) * np.sin(psi), np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi),
     np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)],
    [-1 * np.sin(theta), np.sin(phi) * np.cos(theta), np.cos(phi) * np.cos(theta)]]
    return pd.DataFrame(data)

def main():
    # Parse data
    gps_lla_df = get_data('gps_pos_lla.txt')
    gps_vel_df = get_data('gps_vel_ned.txt')
    imu_df = get_data('imu.txt')
    time_df = get_data('time.txt')



main()