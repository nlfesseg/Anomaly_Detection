from __future__ import absolute_import, division, print_function, unicode_literals

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmdarima as pm
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR

from models.base_model import BaseModel
from util import replace_multiple, invert_transformation


class VarModel(BaseModel):
    def __init__(self, feature, run_id):
        super().__init__(feature, run_id)

    def train(self, feature):
        df_stationary = feature.train_multi
        intersect_ids = np.intersect1d(feature.features_considered_ids, feature.non_stationary_feature_ids)
        print(intersect_ids)
        for id in intersect_ids:
            df_stationary[id] = df_stationary[id].diff().fillna(0)
        # evaluate parameters
        if len(feature.train_multi.columns) > 1:
            best_lag = 0
            temp_model = VAR(endog=df_stationary)
            mae_period = 99999999
            for lag in range(1, 101):
                result = temp_model.fit(maxlags=lag)
                future_forecast_pred = result.forecast(result.endog,
                                                       steps=int(self.config['LSTM_PARAMS']['FUTURE_TARGET']))
                future_forecast_pred = pd.DataFrame(data=future_forecast_pred,
                                                    columns=feature.features_considered_ids)
                future_forecast_pred[intersect_ids] = invert_transformation(feature.train_multi[intersect_ids],
                                                                            future_forecast_pred[intersect_ids])
                future_forecast_pred = future_forecast_pred[feature.id].values
                if not np.isnan(future_forecast_pred).any():
                    mae_temp = mean_absolute_error(feature.val_multi[feature.id].values, future_forecast_pred)
                    if mae_temp < mae_period:
                        best_lag = lag
                        mae_period = mae_temp
            print(best_lag)
            self.model = VAR(endog=feature.train_multi)
            self.model = self.model.fit(maxlags=best_lag)
        else:
            p_values = range(1, 5)
            d_values = range(0, 3)
            q_values = range(0, 3)
            best_score, best_cfg = float("inf"), None
            for p in p_values:
                for d in d_values:
                    for q in q_values:
                        order = (p, d, q)
                        try:
                            model = ARIMA(feature.train_multi, order=order)
                            model_fit = model.fit(disp=0)
                            y_hat = model_fit.forecast(steps=int(self.config['LSTM_PARAMS']['FUTURE_TARGET']))[0]
                            y_hat = y_hat.reshape(-1, 1)
                            mse = mean_squared_error(feature.val_multi, y_hat)
                            if mse < best_score:
                                best_score, best_cfg = mse, order
                            print('ARIMA%s MSE=%.3f' % (order, mse))
                        except:
                            break
            print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
            self.model = ARIMA(feature.train_multi, order=best_cfg)
            self.model = self.model.fit(disp=0)

    def save(self):
        self.model.save(os.path.join('data', self.run_id, 'models', 'VAR',
                                     '{}_VAR.h5'.format(replace_multiple(self.feat_id,
                                                                         ['/', '\\',
                                                                          ':', '?',
                                                                          '*', '"',
                                                                          '<', '>',
                                                                          '|'],
                                                                         "x"))))

    def load(self):
        self.model = tf.keras.models.load_model(os.path.join('../data', self.config['RUNTIME_PARAMS']['USE_ID'],
                                                             'models', 'VAR',
                                                             '{}_VAR.h5'.format(replace_multiple(self.feat_id,
                                                                                                 ['/', '\\',
                                                                                                  ':', '?',
                                                                                                  '*', '"',
                                                                                                  '<', '>',
                                                                                                  '|'],
                                                                                                 "x"))))

    def predict(self, feature):
        if len(feature.val_multi.columns) > 1:
            self.y_hat = self.model.forecast(self.model.endog, steps=int(self.config['LSTM_PARAMS']['FUTURE_TARGET']))
            self.y_hat = pd.DataFrame(data=self.y_hat,
                                      columns=feature.features_considered_ids)
            intersect_ids = np.intersect1d(feature.features_considered_ids, feature.non_stationary_feature_ids)
            self.y_hat[intersect_ids] = invert_transformation(feature.train_multi[intersect_ids],
                                                              self.y_hat[intersect_ids])
        else:
            self.y_hat = self.model.forecast(steps=int(self.config['LSTM_PARAMS']['FUTURE_TARGET']))[0]
            self.y_hat = pd.DataFrame(data=self.y_hat,
                                      columns=feature.features_considered_ids)

        self.y_hat = self.y_hat[self.feat_id]
        feature.y_hat = self.y_hat
        return feature

    def result(self, feature, model):
        pass
