from __future__ import absolute_import, division, print_function, unicode_literals

import configparser
import os

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler

from models.lstm_model import LstmModel
from models.var_model import VarModel
from transform import replace_multiple, adfuller_test, cointegration_test, invert_transformation, pearson, \
    anomaly_detection


class Feature:
    def __init__(self, feature_id, run_id, dataset=None):
        self.id = feature_id
        self.run_id = run_id
        self.config = configparser.ConfigParser()
        self.non_stationary_feature_ids = None
        self.relevant_feature_ids = None
        self.var_model = None

        if dataset is None:
            self.load_values()
        else:
            self.feature_selection(dataset)
            self.feature_analyse(dataset)
            self.save_values()

    def feature_selection(self, dataset):
        self.relevant_feature_ids = pearson(dataset, self.id)
        featured_dataset = dataset[self.relevant_feature_ids]
        if len(self.relevant_feature_ids) > 1:
            dif_featured_dataset = featured_dataset.copy(deep=True)
            dif_featured_dataset = dif_featured_dataset.diff().dropna().round(5)
            dif_featured_dataset = dif_featured_dataset.T.drop_duplicates().T
            self.relevant_feature_ids = list(dif_featured_dataset.columns.values)

    def feature_analyse(self, dataset):
        featured_dataset = dataset[self.relevant_feature_ids]
        self.non_stationary_feature_ids = []
        for name, column in featured_dataset.iteritems():
            col_stationary = adfuller_test(column, name=column.name)
            if not col_stationary:
                self.non_stationary_feature_ids.append(column.name)

        if (len(self.non_stationary_feature_ids) > 1) and (
                len(self.relevant_feature_ids) == len(self.non_stationary_feature_ids)):
            cointegration_test(featured_dataset)

    def train_models(self, dataset):
        dataset = dataset[self.relevant_feature_ids]
        # scaled_dataset = self.scalar.transform(
        #     dataset.values.reshape(-1, len(self.relevant_feature_ids)))
        # scaled_dataset = pd.DataFrame(data=scaled_dataset, columns=self.relevant_feature_ids)

        intersect_ids = np.intersect1d(self.relevant_feature_ids, self.non_stationary_feature_ids)
        for intersect_id in intersect_ids:
            dataset[intersect_id] = dataset[intersect_id].diff()
        dataset = dataset.dropna().reset_index(drop=True)

        self.config.read(os.path.join('temp', self.run_id, 'config', 'config.ini'))
        if int(self.config['RUNTIME_PARAMS']['VAR']):
            self.var_model = VarModel(self.id, self.run_id, dataset=dataset)
        # if int(self.config['RUNTIME_PARAMS']['LSTM']):
        #     self.lstm_model = LstmModel(self.id, self.run_id, dataset=scaled_dataset)

    def load_models(self):
        self.config.read(os.path.join('runs', self.run_id, 'config', 'config.ini'))
        if int(self.config['RUNTIME_PARAMS']['VAR']):
            self.var_model = VarModel(self.id, self.run_id)
        # if int(self.config['RUNTIME_PARAMS']['LSTM']):
        #     self.lstm_model = LstmModel(self.id, self.run_id)

    def predict(self, dataset, view_history):
        dataset = dataset[self.relevant_feature_ids]
        dif_dataset = dataset.copy(deep=True)
        intersect_ids = np.intersect1d(self.relevant_feature_ids, self.non_stationary_feature_ids)
        for intersect_id in intersect_ids:
            dif_dataset[intersect_id] = dif_dataset[intersect_id].diff()
        dif_dataset = dif_dataset.dropna().reset_index(drop=True)

        if view_history < int(self.config['FORECAST_PARAMS']['PAST_HISTORY']):
            view_history = int(self.config['FORECAST_PARAMS']['PAST_HISTORY'])

        history = dataset.iloc[-view_history:]
        actual = dataset.iloc[-int(self.config['FORECAST_PARAMS']['PAST_HISTORY']):]
        results = []
        if int(self.config['RUNTIME_PARAMS']['VAR']):
            var_prediction = self.var_model.predict(dif_dataset)

            if len(self.non_stationary_feature_ids) > 0:
                var_prediction[self.non_stationary_feature_ids] = invert_transformation(
                    dataset[self.non_stationary_feature_ids],
                    var_prediction[self.non_stationary_feature_ids],
                    index=-(int(self.config['FORECAST_PARAMS']['PAST_HISTORY']) + 1))

            var_forecast = var_prediction.iloc[-int(self.config['FORECAST_PARAMS']['FUTURE_TARGET']):]
            anomaly_scores = anomaly_detection(dataset,
                                               pd.concat([history, var_forecast], ignore_index=True), self.id)

            var_prediction = var_prediction[self.id]
            var_forecast = var_prediction.iloc[-int(self.config['FORECAST_PARAMS']['FUTURE_TARGET']):]
            var_prediction = var_prediction.iloc[:int(self.config['FORECAST_PARAMS']['PAST_HISTORY'])]

            var_result = super(VarModel, self.var_model).result(history[self.id].values.flatten(), actual[self.id], var_prediction,
                                                                var_forecast, anomaly_scores, 'VAR')
            results.append(var_result)
        return results

    def save_values(self):
        joblib.dump([self.id, self.relevant_feature_ids, self.non_stationary_feature_ids],
                    os.path.join('temp', self.run_id, 'features',
                                 '{}.pkl'.format(replace_multiple(self.id, ['/', '\\', ':', '?',
                                                                            '*', '"', '<', '>',
                                                                            '|'], "x"))))

    def load_values(self):
        self.id, self.relevant_feature_ids, self.non_stationary_feature_ids = \
            joblib.load(os.path.join('runs', self.run_id, 'features', replace_multiple(
                self.id, ['/', '\\', ':', '?', '*', '"', '<', '>', '|'], "x")))
