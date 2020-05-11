from __future__ import absolute_import, division, print_function, unicode_literals

import configparser
import os
from abc import abstractmethod, ABCMeta

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

from transform import anomaly_detection


class BaseModel(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, feat_id, run_id, dataset=None):
        self.feat_id = feat_id
        self.run_id = run_id
        self.model = None
        self.config = configparser.ConfigParser()

        if dataset is None:
            self.config.read(os.path.join('runs', self.run_id, 'config', 'config.ini'))
            self.load()
        else:
            self.config.read(os.path.join('temp', self.run_id, 'config', 'config.ini'))
            self.train(dataset)
            self.save()

    @abstractmethod
    def train(self, dataset):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def predict(self, dataset):
        pass

    @abstractmethod
    def result(self, history, actual, prediction, forecast, df_aler, model_type):
        mse = mean_squared_error(actual, prediction)
        mae = mean_absolute_error(actual, prediction)
        rmse = np.sqrt(mse)

        df_aler['points'] = df_aler.index

        df_aler = df_aler[df_aler['outlier'] == 1]
        past_alert = df_aler[df_aler['points'] < len(history)]
        future_alert = df_aler[df_aler['points'] >= len(history)]

        output = {'history': history.tolist(), 'expected': prediction.tolist(), 'forecast': forecast.tolist(),
                  'rmse': rmse, 'mse': mse, 'mae': mae, 'future_alerts': future_alert.fillna(0).to_dict(orient='record'),
                  'past_alerts': past_alert.fillna(0).to_dict(orient='record'), 'model': model_type}
        # var_output['future'] = df_result_forecast.fillna(0).to_dict(orient='record')
        return output
