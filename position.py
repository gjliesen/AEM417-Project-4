import numpy as np


def extract_data(df, time):
    df = df.loc[time, ['phi', 'theta', 'psi']]
    df = df.to_numpy().reshape((-1, 1))
    return df


def pe_matrix(rn, re, h, lat):
    return np.array([[1 / (rn + h), 0, 0],
                     [0, 1 / ((re + h) * np.cos(lat)), 0],
                     [0, 0, -1]])


def update(rn, re, h, lat, p_e, v_n, dt):
    pe_mat = pe_matrix(rn, re, h, lat)
    return p_e + dt * pe_mat @ v_n

