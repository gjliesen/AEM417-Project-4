import pandas as pd
import numpy as np
import constants as cn
import attitude
import velocity
import position


def calc_wn_in(wn_ie, wn_en):
    return wn_ie + wn_en


def calc_wb_nb(wn_in, wb_ib):
    return wb_ib - wn_in + cn.gyro_bias


def noise_gain_matrix():
    print('Placeholder')


def loose_state_matrix(pos_df, vel_df, ins_df, cur, att_cur, v_n_cur, pos_cur, psi_nb):
    pos_gps = position.extract_data(pos_df, cur)
    vel_gps = velocity.extract_data(vel_df, cur)
    del_x = np.array([[pos_gps - pos_cur],
                      [vel_gps - v_n_cur],
                      [psi_nb]])

    print('placeholder')
