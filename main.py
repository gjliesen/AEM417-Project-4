import pandas as pd
import numpy as np
import math as m

def get_data(file):
    df = pd.read_csv(file, index_col=False, squeeze=True)
    return df


def main():
    # Parse data
    gps_lla_df = get_data('gps_pos_lla.txt')
    gps_vel_df = get_data('gps_vel_ned.txt')
    imu_df = get_data('imu.txt')
    time_df = get_data('time.txt')



main()