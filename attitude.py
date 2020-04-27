import numpy as np


def extract_data(df, time):
    df = df.loc[time, ['phi', 'theta', 'psi']]
    df = df.to_numpy().reshape((-1, 1))
    return df


def calc_a_nb_matrix(phi, theta):
    a_nb = np.array([[1, np.sin(phi) * np.sin(theta), np.cos(phi) * np.sin(theta)],
                     [0, np.cos(phi) * np.cos(theta), -1 * np.sin(phi) * np.cos(theta)],
                     [0, np.sin(phi), np.cos(phi)]])
    a_nb *= (1 / np.cos(theta))
    return a_nb


def get(phi, theta, wb_ib, c_bn, psi_nb, wn_ie, wn_en, dt):
    wb_in = c_bn @ (wn_ie + wn_en)
    a_nb = calc_a_nb_matrix(phi, theta)
    wb_nb = wb_ib - wb_in
    return (psi_nb + dt * a_nb @ wb_nb).T
