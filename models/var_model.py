from __future__ import absolute_import, division, print_function, unicode_literals

import os

import joblib
import pandas as pd
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.vector_ar.var_model import VAR

from models.base_model import BaseModel
from transform import replace_multiple


class VarModel(BaseModel):
    def __init__(self, feat_id, run_id, dataset=None):
        self.optimal_p = 1
        super().__init__(feat_id, run_id, dataset)

    def train(self, dataset):
        if len(dataset.columns) > 1:
            self.model = VAR(dataset)
            self.optimal_p = self.model.select_order(20).aic
        else:
            self.model = AR(dataset)
            self.optimal_p = self.model.select_order(20, 'aic')

    def save(self):
        joblib.dump(self.optimal_p,
                    os.path.join('temp', self.run_id, 'models', 'VAR',
                                 '{}_VAR.pkl'.format(replace_multiple(self.feat_id,
                                                                      ['/', '\\',
                                                                       ':', '?',
                                                                       '*', '"',
                                                                       '<', '>',
                                                                       '|'],
                                                                      "x"))))

    def load(self):
        self.optimal_p = \
            joblib.load(os.path.join('runs', self.run_id, 'models', 'VAR',
                                     '{}_VAR.pkl'.format(replace_multiple(self.feat_id,
                                                                          ['/', '\\',
                                                                           ':', '?',
                                                                           '*', '"',
                                                                           '<', '>',
                                                                           '|'],
                                                                          "x"))))

    def result(self, history, actual, prediction, forecast, df_aler, model_type):
        pass

    def predict(self, dataset):
        start_idx = len(dataset) - int(self.config['FORECAST_PARAMS']['PAST_HISTORY'])
        end_idx = len(dataset) + int(self.config['FORECAST_PARAMS']['FUTURE_TARGET']) - 1

        if len(dataset.columns) > 1:
            self.model = VAR(dataset)
            result = self.model.fit(self.optimal_p)
            prediction = self.model.predict(result.params, start=start_idx, end=end_idx, lags=self.optimal_p)
            return pd.DataFrame(data=prediction, columns=dataset.columns.values)
        else:
            self.model = AR(dataset)
            self.model = self.model.fit(self.optimal_p)
            prediction = self.model.predict(start=start_idx, end=end_idx)
            return pd.DataFrame(data=prediction, columns=dataset.columns.values)
