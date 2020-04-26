import pandas as pd
import ins_formulation as ins
# import numpy as np
# import constants as cn


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


def initialize_ins_data(imu_df, pos_df, vel_df):
    columns = ['Time', 'dt', 'lat', 'long', 'h', 'phi', 'theta', 'psi', 'vN', 'vE', 'vD']
    ins_df = pd.DataFrame(index=imu_df.index, columns=columns)
    ins_df.Time = imu_df.Time
    ins_df.dt = ins_df.Time - ins_df.Time.shift(1)
    ins_df = pd.merge(ins_df, imu_df, on='Time')
    ins_df.set_index('Time', inplace=True)

    ins_df.lat.iat[0] = pos_df.lat.iat[0]
    ins_df.long.iat[0] = pos_df.long.iat[0]
    ins_df.h.iat[0] = pos_df.h.iat[0]
    ins_df.vN.iat[0] = vel_df.vN.iat[0]
    ins_df.vE.iat[0] = vel_df.vE.iat[0]
    ins_df.vD.iat[0] = vel_df.vD.iat[0]
    ins_df.iloc[0, 4:7] = 0
    return ins_df


def main():
    # Parse data

    pos_df = get_pos_data('gps_pos_lla.txt')
    vel_df = get_vel_data('gps_vel_ned.txt')
    imu_df = get_imu_data('imu.txt')
    # time_df = get_time_data('time.txt')
    ins_df = initialize_ins_data(imu_df, pos_df, vel_df)
    ins_df = ins.ins_formulation(ins_df)


main()
