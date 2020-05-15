from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.vector_ar.var_model import VAR

from models.base_model import BaseModel
from transform import replace_multiple


class VarModel(BaseModel):
    def __init__(self, feat_id, run_id, dataset=None):
        self.model_type = 'VAR'
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

    def result(self, history, actual, prediction, forecast, df_aler):
        mse = mean_squared_error(actual, prediction)
        mae = mean_absolute_error(actual, prediction)
        rmse = np.sqrt(mse)

        df_aler['points'] = df_aler.index

        df_aler = df_aler[df_aler['outlier'] == 1]
        past_alert = df_aler[df_aler['points'] < len(history)]
        future_alert = df_aler[df_aler['points'] >= len(history)]

        output = {'history': history.tolist(), 'expected': prediction.tolist(), 'forecast': forecast.tolist(),
                  'rmse': rmse, 'mse': mse, 'mae': mae, 'future_alerts': future_alert.fillna(0).to_dict(orient='record'),
                  'past_alerts': past_alert.fillna(0).to_dict(orient='record'), 'model': self.model_type}
        # var_output['future'] = df_result_forecast.fillna(0).to_dict(orient='record')
        return output

    def predict(self, dataset, start_idx, end_idx):
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
