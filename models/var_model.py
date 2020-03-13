from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from statsmodels.tsa.vector_ar.var_model import VAR

from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
import os

from models.base_model import BaseModel
from util import replace_multiple


class VarModel(BaseModel):
    def __init__(self, feature, run_id):
        self.best_lag = None
        super().__init__(feature, run_id)

    def train(self, feature):
        temp_model = VAR(endog=feature.train_multi)
        mae_period = 99999999
        for lag in range(1, int(self.config['LSTM_PARAMS']['PAST_HISTORY'])):
            result = temp_model.fit(maxlags=lag)
            future_forecast_pred = result.forecast(result.endog, steps=len(feature.val_multi))
            future_forecast_pred = pd.DataFrame(data=future_forecast_pred,
                                                columns=feature.features_considered_ids)
            future_forecast_pred = future_forecast_pred[feature.id].values
            if not np.isnan(future_forecast_pred).any():
                mae_temp = mean_absolute_error(feature.val_multi[feature.id].values, future_forecast_pred)
                if mae_temp < mae_period:
                    self.best_lag = lag
                    mae_period = mae_temp
        self.model = VAR(endog=feature.train_multi)
        print(self.best_lag)
        self.model = self.model.fit(maxlags=self.best_lag)

    def save(self):
        self.model.save(os.path.join('data', self.run_id, 'models',
                                     '{}_VAR.h5'.format(replace_multiple(self.feat_id,
                                                                         ['/', '\\',
                                                                          ':', '?',
                                                                          '*', '"',
                                                                          '<', '>',
                                                                          '|'],
                                                                         "x"))))

    def load(self):
        self.model = tf.keras.models.load_model(os.path.join('../data', self.config['RUNTIME_PARAMS']['USE_ID'],
                                                             'models', '{}_VAR.h5'.format(replace_multiple(self.feat_id,
                                                                                                           ['/', '\\',
                                                                                                            ':', '?',
                                                                                                            '*', '"',
                                                                                                            '<', '>',
                                                                                                            '|'],
                                                                                                           "x"))))

    def predict(self, feature):
        self.y_pred = self.model.forecast(self.model.endog, steps=len(feature.val_multi))
        self.y_pred = pd.DataFrame(data=self.y_pred,
                                   columns=feature.features_considered_ids)
        self.y_pred.reset_index()
        feature.y_pred = self.y_pred
        return feature

    def result(self, feature, model):
        pass
