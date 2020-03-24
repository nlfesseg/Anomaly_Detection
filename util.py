from __future__ import absolute_import, division, print_function, unicode_literals

import configparser
import os

import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i:i + target_size])

    return np.array(data), np.array(labels)


def make_dirs(_id):
    config = configparser.ConfigParser()
    config.read('config.ini')

    if not config['RUNTIME_PARAMS'].getboolean('TRAIN') or not config['RUNTIME_PARAMS'].getboolean('PREDICT'):
        if not os.path.isdir('data/%s' % config['RUNTIME_PARAMS']['USE_ID']):
            raise ValueError("Run ID {} is not valid. Provide a valid ID when loading prior models.".format(_id))

    paths = ['data', 'data/%s' % _id, 'data/%s/models' % _id, 'data/%s/scalars' % _id, 'data/%s/models/VAR' % _id,
             'data/%s/models/LSTM' % _id]

    for p in paths:
        if not os.path.isdir(p):
            os.mkdir(p)


def create_time_steps(length):
    return list(range(-length, 0))


def replace_multiple(main_string, replaces, new_string):
    for elem in replaces:
        if elem in main_string:
            main_string = main_string.replace(elem, new_string)

    return main_string


def adfuller_test(series, signif=0.05, name='', verbose=False):
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic': round(r[0], 4), 'pvalue': round(r[1], 4), 'n_lags': round(r[2], 4), 'n_obs': r[3]}
    p_value = output['pvalue']

    def adjust(val, length=6):
        return str(val).ljust(length)

    if p_value <= signif:
        return True
    else:
        return False


def invert_transformation(df_train, df_forecast, dif_times=1):
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:
        # Roll back 2nd Diff
        if dif_times == 2:
            df_fc[col] = (df_train[col].iloc[-1] - df_train[col].iloc[-2]) + df_fc[col].cumsum()
        # Roll back 1st Diff
        df_fc[col] = df_train[col].iloc[-1] + df_fc[col].cumsum()
    return df_fc


def cointegration_test(df, alpha=0.05):
    out = coint_johansen(df, -1, 4)
    d = {'0.90': 0, '0.95': 1, '0.99': 2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1 - alpha)]]

    def adjust(val, length=6): return str(val).ljust(length)

    # Summary
    print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--' * 20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace, 2), 9), ">", adjust(cvt, 8), ' =>  ', trace > cvt)
