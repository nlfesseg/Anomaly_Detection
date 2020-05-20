from __future__ import absolute_import, division, print_function, unicode_literals

import configparser
import os

import joblib
import numpy as np
import pandas as pd
from models.dbscan_model import DbscanModel
from models.var_model import VarModel
from transform import replace_multiple, adfuller_test, cointegration_test, invert_transformation, pearson


class Feature:
    def __init__(self, feature_id, run_id, data=None):
        self.id = feature_id
        self.run_id = run_id
        self.config = configparser.ConfigParser()
        self.non_stat_feat_ids = None
        self.relevant_feat_ids = None
        self.var_model = None
        self.dbscan_model = None

        if data is None:
            self.load_values()
        else:
            self.feature_selection(data)
            self.feature_analyse(data)
            self.save_values()

    def feature_selection(self, data):
        self.relevant_feat_ids = pearson(data, self.id)
        df_feat = data[self.relevant_feat_ids]
        if len(self.relevant_feat_ids) > 1:
            df_feat_dif = df_feat.copy(deep=True)
            df_feat_dif = df_feat_dif.diff().dropna().round(5)
            df_feat_dif = df_feat_dif.T.drop_duplicates().T
            self.relevant_feat_ids = list(df_feat_dif.columns.values)

    def feature_analyse(self, data):
        df_feat = data[self.relevant_feat_ids]
        self.non_stat_feat_ids = []
        for name, column in df_feat.iteritems():
            col_stat = adfuller_test(column, name=column.name)
            if not col_stat:
                self.non_stat_feat_ids.append(column.name)

        if (len(self.non_stat_feat_ids) > 1) and (
                len(self.relevant_feat_ids) == len(self.non_stat_feat_ids)):
            cointegration_test(df_feat)

    def train_models(self, data):
        df = data[self.relevant_feat_ids]

        intersect_ids = np.intersect1d(self.relevant_feat_ids, self.non_stat_feat_ids)
        for intersect_id in intersect_ids:
            df[intersect_id] = df[intersect_id].diff()
        df = df.dropna().reset_index(drop=True)

        self.config.read(os.path.join('temp', self.run_id, 'config', 'config.ini'))
        if int(self.config['RUNTIME_PARAMS']['VAR']):
            self.var_model = VarModel(self.id, self.run_id, data=df)
        self.dbscan_model = DbscanModel(self.id, self.run_id, data=df)
        # if int(self.config['RUNTIME_PARAMS']['LSTM']):
        #     self.lstm_model = LstmModel(self.id, self.run_id, dataset=scaled_dataset)

    def load_models(self):
        self.config.read(os.path.join('runs', self.run_id, 'config', 'config.ini'))
        if int(self.config['RUNTIME_PARAMS']['VAR']):
            self.var_model = VarModel(self.id, self.run_id)
        self.dbscan_model = DbscanModel(self.id, self.run_id)
        # if int(self.config['RUNTIME_PARAMS']['LSTM']):
        #     self.lstm_model = LstmModel(self.id, self.run_id)

    def predict(self, df, past_history, future_target, view_history, cluster_history):
        df = df[self.relevant_feat_ids]
        df_dif = df.copy(deep=True)
        intersect_ids = np.intersect1d(self.relevant_feat_ids, self.non_stat_feat_ids)
        for intersect_id in intersect_ids:
            df_dif[intersect_id] = df_dif[intersect_id].diff()
        df_dif = df_dif.dropna().reset_index(drop=True)

        df_view_hist = df.iloc[-view_history:]
        df_clust_hist = df.iloc[-cluster_history:]
        actual = df.iloc[-past_history:]
        results = []

        if int(self.config['RUNTIME_PARAMS']['VAR']):
            start_idx = len(df) - past_history
            end_idx = len(df) + future_target - 1

            var_pred = self.var_model.predict(df_dif, start_idx, end_idx)

            if len(self.non_stat_feat_ids) > 0:
                var_pred[self.non_stat_feat_ids] = invert_transformation(
                    df[self.non_stat_feat_ids],
                    var_pred[self.non_stat_feat_ids],
                    index=-(past_history + 1))

            fc_var = var_pred.iloc[-future_target:]
            anomaly_scores = self.dbscan_model.predict(pd.concat([df_clust_hist, fc_var], ignore_index=True))
            anomaly_scores = anomaly_scores.iloc[-(view_history + future_target):]
            anomaly_scores.reset_index(inplace=True, drop=True)

            var_pred = var_pred[self.id]
            fc_var = var_pred.iloc[-future_target:]
            var_pred = var_pred.iloc[:past_history]

            result_var = self.var_model.result(df_view_hist[self.id].values.flatten(), actual[self.id],
                                               var_pred, fc_var, anomaly_scores)
            results.append(result_var)
        return results

    def save_values(self):
        joblib.dump([self.id, self.relevant_feat_ids, self.non_stat_feat_ids],
                    os.path.join('temp', self.run_id, 'features',
                                 '{}.pkl'.format(replace_multiple(self.id, ['/', '\\', ':', '?',
                                                                            '*', '"', '<', '>',
                                                                            '|'], "x"))))

    def load_values(self):
        self.id, self.relevant_feat_ids, self.non_stat_feat_ids = \
            joblib.load(os.path.join('runs', self.run_id, 'features', replace_multiple(
                self.id, ['/', '\\', ':', '?', '*', '"', '<', '>', '|'], "x")))
