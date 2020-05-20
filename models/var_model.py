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
    def __init__(self, feat_id, run_id, data=None):
        self.model_type = 'VAR'
        self.opt_p = 1
        super().__init__(feat_id, run_id, data)

    def train(self, data):
        if len(data.columns) > 1:
            self.model = VAR(data)
            self.opt_p = self.model.select_order(30).aic
        else:
            self.model = AR(data)
            self.opt_p = self.model.select_order(30, 'aic')

    def save(self):
        joblib.dump(self.opt_p,
                    os.path.join('temp', self.run_id, 'models', 'VAR',
                                 '{}_VAR.pkl'.format(replace_multiple(self.feat_id,
                                                                      ['/', '\\',
                                                                       ':', '?',
                                                                       '*', '"',
                                                                       '<', '>',
                                                                       '|'],
                                                                      "x"))))

    def load(self):
        self.opt_p = \
            joblib.load(os.path.join('runs', self.run_id, 'models', 'VAR',
                                     '{}_VAR.pkl'.format(replace_multiple(self.feat_id,
                                                                          ['/', '\\',
                                                                           ':', '?',
                                                                           '*', '"',
                                                                           '<', '>',
                                                                           '|'],
                                                                          "x"))))

    def result(self, history, actual, prediction, forecast, anomaly_scores):
        mse = mean_squared_error(actual, prediction)
        mae = mean_absolute_error(actual, prediction)
        rmse = np.sqrt(mse)

        anomaly_scores['points'] = anomaly_scores.index

        future_alert = anomaly_scores.tail(len(forecast))
        past_alert = anomaly_scores.iloc[:len(history)]
        future_alert = future_alert[future_alert['outlier'] == -1]
        past_alert = past_alert[past_alert['outlier'] == -1]

        output = {'history': history.tolist(), 'expected': prediction.tolist(), 'forecast': forecast.tolist(),
                  'rmse': rmse, 'mse': mse, 'mae': mae, 'future_alerts': future_alert.fillna(0).to_dict(orient='record'),
                  'past_alerts': past_alert.fillna(0).to_dict(orient='record'), 'model': self.model_type}
        return output

    def predict(self, data, start_idx, end_idx):
        if len(data.columns) > 1:
            self.model = VAR(data)
            result = self.model.fit(self.opt_p)
            y_pred = self.model.predict(result.params, start=start_idx, end=end_idx, lags=self.opt_p)
            return pd.DataFrame(data=y_pred, columns=data.columns.values)
        else:
            self.model = AR(data)
            self.model = self.model.fit(self.opt_p)
            y_pred = self.model.predict(start=start_idx, end=end_idx)
            return pd.DataFrame(data=y_pred, columns=data.columns.values)
