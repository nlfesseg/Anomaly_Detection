from __future__ import absolute_import, division, print_function, unicode_literals

import configparser
import os

import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import MinMaxScaler

from util import multivariate_data, replace_multiple, adfuller_test, cointegration_test


class Feature:
    def __init__(self, feature_id, run_id, dataset):
        self.id = feature_id
        self.run_id = run_id
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
        self.scalar = None
        self.train_multi = None
        self.val_multi = None
        self.x_train_multi = None
        self.df_differenced = None
        self.non_stationary_feature_ids = None
        self.y_train_multi = None
        self.x_val_multi = None
        self.y_val_multi = None
        self.x_val_multi_split = None
        self.y_val_multi_split = None
        self.y_hat = None
        self.features_considered_ids = None

        if not self.config['RUNTIME_PARAMS'].getboolean('TRAIN'):
            try:
                self.load_values()
            except FileNotFoundError:
                self.fit_scalar(dataset)
                self.feature_selection(dataset)
                self.save_values()

        else:
            self.fit_scalar(dataset)
            self.feature_selection(dataset)
            self.save_values()

    def shape_data(self, dataset, train=True):
        column_loc = dataset.columns.get_loc(self.id)
        # self.train_multi = dataset.iloc[:-(int(int(self.config['LSTM_PARAMS']['FUTURE_TARGET']) / 2))]

        # # cointegration test
        # cointegration_test(self.df_differenced)

        # self.val_multi = dataset.iloc[-(int(int(self.config['LSTM_PARAMS']['FUTURE_TARGET']) / 2)):]
        # self.val_multi.index = np.arange(len(self.val_multi))
        dataset = dataset.values
        self.x_train_multi, self.y_train_multi = multivariate_data(dataset, dataset[:, column_loc], 0,
                                                                   int(self.config['LSTM_PARAMS']['TRAIN_SPLIT']),
                                                                   int(self.config['LSTM_PARAMS']['PAST_HISTORY']),
                                                                   int(self.config['LSTM_PARAMS']['FUTURE_TARGET']),
                                                                   int(self.config['LSTM_PARAMS']['STEP']))
        self.x_val_multi, self.y_val_multi = multivariate_data(dataset, dataset[:, column_loc],
                                                               int(self.config['LSTM_PARAMS']['TRAIN_SPLIT']), None,
                                                               int(self.config['LSTM_PARAMS']['PAST_HISTORY']),
                                                               int(self.config['LSTM_PARAMS']['FUTURE_TARGET']),
                                                               int(self.config['LSTM_PARAMS']['STEP']))
        start_idx = int(self.config['LSTM_PARAMS']['TRAIN_SPLIT']) + (len(self.x_val_multi) + 1)
        self.x_val_multi_split, self.y_val_multi_split = multivariate_data(dataset, dataset[:, column_loc],
                                                                           start_idx, (len(dataset) + 1),
                                                                           int(self.config['LSTM_PARAMS'][
                                                                                   'PAST_HISTORY']),
                                                                           int(self.config['LSTM_PARAMS'][
                                                                                   'FUTURE_TARGET']),
                                                                           int(self.config['LSTM_PARAMS']['STEP']))

    def fit_scalar(self, dataset):
        train_data = dataset[:int(self.config['LSTM_PARAMS']['TRAIN_SPLIT'])]
        train_data = train_data[self.id]
        self.scalar = MinMaxScaler(feature_range=(0, 1))
        self.scalar.fit(train_data.values.reshape(-1, 1))

    def feature_selection(self, dataset):
        x = dataset.drop(self.id, axis=1)  # Feature Matrix
        y = dataset[self.id]  # Target Variable
        reg = LassoCV()
        reg.fit(x, y)
        print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
        print("Best score using built-in LassoCV: %f" % reg.score(x, y))
        coef = pd.Series(reg.coef_, index=x.columns)

        print(
            "Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +
            str(sum(coef == 0)) + " variables")

        self.features_considered_ids = list(coef[coef != 0].index)
        self.features_considered_ids = list()
        self.features_considered_ids.insert(0, y.name)
        featured_dataset = dataset[self.features_considered_ids]
        if len(self.features_considered_ids) > 1:
            featured_dataset = featured_dataset.T.drop_duplicates().T
        self.features_considered_ids = list(featured_dataset.columns.values)

        self.non_stationary_feature_ids = []
        for name, column in featured_dataset.iteritems():
            col_stationary = adfuller_test(column, name=column.name)
            if not col_stationary:
                self.non_stationary_feature_ids.append(column.name)

    def save_values(self):
        joblib.dump([self.scalar, self.features_considered_ids, self.non_stationary_feature_ids],
                    os.path.join('data', self.run_id, 'scalars',
                                 '{}.pkl'.format(replace_multiple(self.id,
                                                                  ['/',
                                                                   '\\',
                                                                   ':',
                                                                   '?',
                                                                   '*',
                                                                   '"',
                                                                   '<',
                                                                   '>',
                                                                   '|'],
                                                                  "x"))))

    def load_values(self):
        self.scalar, self.features_considered_ids, self.non_stationary_feature_ids = \
            joblib.load(os.path.join('data',
                                     self.config[
                                         'RUNTIME_PARAMS'][
                                         'USE_ID'],
                                     'scalars',
                                     replace_multiple(
                                         self.id,
                                         ['/',
                                          '\\',
                                          ':',
                                          '?',
                                          '*',
                                          '"',
                                          '<',
                                          '>',
                                          '|'],
                                         "x") + '.pkl'))

    def load_data(self):
        pass
