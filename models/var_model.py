from __future__ import absolute_import, division, print_function, unicode_literals

import os

import numpy as np
import pandas as pd
import tensorflow as tf
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.vector_ar.var_model import VAR

from models.base_model import BaseModel
from util import replace_multiple, invert_transformation


class VarModel(BaseModel):
    def __init__(self, feature, run_id):
        super().__init__(feature, run_id)

    def train(self, feature):
        train_differenced = feature.df_differenced.iloc[:(int(self.config['LSTM_PARAMS']['TRAIN_SPLIT'])
                                                          + int(self.config['LSTM_PARAMS']['PAST_HISTORY']))]
        if len(train_differenced.columns) > 1:
            self.model = VAR(endog=train_differenced)
        else:
            self.model = AR(endog=train_differenced)
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
        train_differenced = np.array(feature.df_differenced.iloc[:(int(self.config['LSTM_PARAMS']['TRAIN_SPLIT'])
                                                                   + int(self.config['LSTM_PARAMS']['PAST_HISTORY']))])
        train_data = feature.train_multi
        if len(feature.val_multi.columns) > 1:
            # for i in range(len(feature.train_multi), len(feature.df_differenced)):
            #     # Save the new set of forecasts
            #     forecast = \
            #         self.model.forecast(train_differenced, steps=int(self.config['LSTM_PARAMS']['FUTURE_TARGET']))[0]
            #     forecast = forecast.reshape(1, -1)
            #     self.y_pred = np.concatenate((self.y_pred, forecast), axis=0)
            #     # Update the results by appending the next observation
            #     updated_endog = feature.df_differenced.iloc[i:i + 1]
            #     train_differenced = np.concatenate((train_differenced, updated_endog), axis=0)
            # Save the new set of forecasts
            forecast = \
                self.model.forecast(train_differenced, steps=len(feature.val_multi))
            # forecast = forecast.reshape(1, -1)
            self.y_pred = np.concatenate((self.y_pred, forecast), axis=0)
        else:
            for i in range(len(feature.train_multi), len(feature.df_differenced)):
                # Save the new set of forecasts
                forecast = self.model.predict(start=len(train_differenced), end=(
                        len(train_differenced) + int(self.config['LSTM_PARAMS']['FUTURE_TARGET'])))[0]
                self.y_pred = np.concatenate((self.y_pred, forecast), axis=0)
                # Update the results by appending the next observation
                updated_endog = feature.df_differenced.iloc[i:i + 1]
                train_differenced = np.concatenate((train_differenced, updated_endog), axis=0)

        self.y_pred = pd.DataFrame(data=self.y_pred,
                                   columns=feature.features_considered_ids)
        if feature.dif_times >= 1:
            for i in range(0, len(self.y_pred)):
                self.y_pred.iloc[i] = invert_transformation(train_data, self.y_pred.iloc[i],
                                                            dif_times=feature.dif_times)
                train_data = train_data.append(feature.val_multi.iloc[i])

        feature.y_pred = self.y_pred
        return feature

    def result(self, feature, model):
        pass
