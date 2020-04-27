import pandas as pd
import ins
import constants as cn
import matplotlib.pyplot as plt
import navpy


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
    columns = ['Time', 'dt', 'lat', 'long', 'h', 'phi', 'theta', 'psi', 'vN', 'vE', 'vD', 'acc_bias', 'gyro_bias']
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
    ins_df.acc_bias.iat[0] = cn.acc_bias
    ins_df.gyro_bias.iat[0] = cn.gyro_bias
    ins_df.iloc[0, 4:7] = 0
    return ins_df


def get_ned_pos(ins_df, imu_df):
    ned_pos = pd.DataFrame(data=None, index=ins_df.index,
                           columns=['X', 'Y', 'Z'])
    for time in ins_df.index:
        NED = navpy.lla2ned(ins_df.loc[(time, 'lat')],
                            ins_df.loc[(time, 'long')],
                            ins_df.loc[(time, 'h')],
                            cn.pos_ref[0][0],
                            cn.pos_ref[1][0],
                            cn.pos_ref[2][0],
                            latlon_unit='rad', alt_unit='m', model='wgs84')
        ned_pos.loc[(time, 'X')] = NED[0]
        ned_pos.loc[(time, 'Y')] = NED[1]
        ned_pos.loc[(time, 'Z')] = NED[2]
        if time == 424:
            break
    return ned_pos


def main():
    # Parse data
    imu_df = get_imu_data('imu.txt')
    pos_df = get_pos_data('gps_pos_lla.txt')
    pos_df.index = imu_df.Time
    vel_df = get_vel_data('gps_vel_ned.txt')
    vel_df.index = imu_df.Time
    ins_df = initialize_ins_data(imu_df, pos_df, vel_df)
    ins_df = ins.solve(ins_df, pos_df, vel_df)
    ins_df = ins_df.drop(ins_df.loc[424::].index)
    ned_pos = get_ned_pos(ins_df, imu_df)

    plt.plot(ned_pos.X, ned_pos.Y)
    plt.xlabel('North')
    plt.ylabel('East')
    plt.title('Trajectory')
    plt.show()

    plt.plot(ned_pos.Z)
    plt.xlabel('Time')
    plt.ylabel('Meters')
    plt.title('Height')
    plt.show()

    plt.plot(ins_df.vN)
    plt.plot(ins_df.vE)
    plt.plot(ins_df.vD)
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    plt.title('NED Velocity')
    plt.legend(['X', 'Y', 'Z'])
    plt.show()

    plt.plot(ins_df.phi)
    plt.plot(ins_df.theta)
    plt.plot(ins_df.psi)
    plt.xlabel('Time')
    plt.ylabel('Radians')
    plt.title('Euler Angles')
    plt.legend(['Phi', 'Theta', 'Psi'])
    plt.show()
    plt.plot(ins_df.acc_bias[0][0].flatten())
    plt.plot(ins_df.acc_bias[1][0].flatten())
    plt.plot(ins_df.acc_bias[2][0].flatten())
    plt.xlabel('Time')
    plt.ylabel('Bias')
    plt.title('Accelerometer Bias')
    plt.legend(['X', 'Y', 'Z'])
    plt.show()

    plt.plot(ins_df.acc_bias[0][0])
    plt.plot(ins_df.acc_bias[1][0])
    plt.plot(ins_df.acc_bias[2][0])
    plt.xlabel('Time')
    plt.ylabel('Bias')
    plt.title('Gyroscope Bias')
    plt.legend(['X', 'Y', 'Z'])
    plt.show()


main()
