import numpy as np
import constants as cn


def extract_data(df, time):
    df = df.loc[time, ['vN', 'vE', 'vD']]
    df = df.to_numpy().reshape((-1, 1))
    return df


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


def get(ins_df, prev, h, lat, wn_ie, wn_en, c_bn, v_n, dt, acc_bias):
    f_b = ins_df.loc[prev, ['aX', 'aY', 'aZ']]
    f_b = f_b.to_numpy().reshape((-1, 1))
    f_b = f_b + acc_bias
    g_n = g_n_matrix(lat, h)
    f_n = c_bn @ f_b
    v_n_dot = f_n + g_n - cn.get_skew((2 * wn_ie - wn_en)) @ v_n
    return v_n + dt * v_n_dot
