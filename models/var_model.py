from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from sklearn.externals import joblib
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.vector_ar.var_model import VAR

from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
import os

from models.base_model import BaseModel
from util import replace_multiple, adfuller_test, invert_transformation


class VarModel(BaseModel):
    def __init__(self, feature, run_id):
        super().__init__(feature, run_id)

    def train(self, feature):
        # train_differenced = feature.df_differenced.iloc[:(int(self.config['LSTM_PARAMS']['TRAIN_SPLIT'])
        #                                                   + int(self.config['LSTM_PARAMS']['PAST_HISTORY']))]
        if len(feature.train_multi.columns) > 1:
            self.model = VAR(endog=feature.train_multi)
        else:
            self.model = AR(endog=feature.train_multi)
        self.model = self.model.fit()

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
        self.y_pred = np.empty((0, len(feature.val_multi.columns)), int)
        # train_differenced = np.array(feature.df_differenced.iloc[:(int(self.config['LSTM_PARAMS']['TRAIN_SPLIT'])
        #                                                   + int(self.config['LSTM_PARAMS']['PAST_HISTORY']))])
        train_data = np.array(feature.train_multi)
        if len(feature.val_multi.columns) > 1:
            for i in range(0, len(feature.val_multi)):
                # Save the new set of forecasts
                forecast = \
                    self.model.forecast(train_data, steps=int(self.config['LSTM_PARAMS']['FUTURE_TARGET']))[0]
                forecast = forecast.reshape(1, -1)
                self.y_pred = np.concatenate((self.y_pred, forecast), axis=0)
                # Update the results by appending the next observation
                updated_endog = feature.val_multi.iloc[i:i + 1]
                train_data = np.concatenate((train_data, updated_endog), axis=0)
        else:
            for i in range(0, len(feature.val_multi)):
                # Save the new set of forecasts
                forecast = self.model.predict(start=len(train_data), end=(
                        len(train_data) + int(self.config['LSTM_PARAMS']['FUTURE_TARGET'])))[0]
                self.y_pred = np.concatenate((self.y_pred, forecast), axis=0)
                # Update the results by appending the next observation
                updated_endog = feature.val_multi.iloc[i:i + 1]
                train_differenced = np.concatenate((train_data, updated_endog), axis=0)

        self.y_pred = pd.DataFrame(data=self.y_pred,
                                   columns=feature.features_considered_ids)
        self.y_pred.reset_index()
        if feature.dif_times >= 1:
            self.y_pred = invert_transformation(feature.train_multi, self.y_pred, dif_times=feature.dif_times)

        feature.y_pred = self.y_pred
        return feature

    def result(self, feature, model):
        pass
