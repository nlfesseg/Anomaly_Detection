from __future__ import absolute_import, division, print_function, unicode_literals

import configparser
from abc import abstractmethod, ABCMeta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.losses import mean_absolute_error, mean_squared_error

from plotter import multi_step_plot


class BaseModel(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, feature, run_id):
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
        self.feat_id = feature.id
        self.run_id = run_id
        self.model = None
        self.y_hat = np.array([])

        if not self.config['RUNTIME_PARAMS'].getboolean('PREDICT'):
            try:
                self.load()
            except FileNotFoundError:
                self.train(feature)
                # self.save()

        else:
            self.train(feature)
            # self.save()

    @abstractmethod
    def train(self, feature):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def predict(self, feature):
        pass

    @abstractmethod
    def result(self, feature, model_type):
        mse = mean_squared_error(feature.val_multi[feature.id],
                                 self.y_hat[feature.id].iloc[:int(int(self.config['LSTM_PARAMS']['FUTURE_TARGET']) / 2)])
        mae = mean_absolute_error(feature.val_multi[feature.id],
                                  self.y_hat[feature.id].iloc[:int(int(self.config['LSTM_PARAMS']['FUTURE_TARGET']) / 2)])
        rmse = np.sqrt(mse)

        history = feature.train_multi.iloc[-int(self.config['LSTM_PARAMS']['PAST_HISTORY']):]
        history = feature.scalar.inverse_transform(history)

        # true_future = np.array(true_future).reshape(-1, 1)
        true_future = feature.scalar.inverse_transform(feature.val_multi)

        # prediction = np.array(prediction).reshape(-1, 1)
        prediction = feature.scalar.inverse_transform(self.y_hat)

        multi_step_plot(history, true_future, prediction)

        df_aler = pd.DataFrame()
        df_aler['real_value'] = feature.val_multi[feature.id]
        df_aler['expected value'] = self.y_hat[feature.id].iloc[:-int(self.config['LSTM_PARAMS']['FUTURE_TARGET'])]
        df_aler['mse'] = mse
        df_aler['points'] = self.y_hat.index
        df_aler.set_index('points', inplace=True)
        df_aler['mae'] = mae

        df_aler['anomaly_score'] = abs(df_aler['expected value'] - df_aler['real_value']) / df_aler['mae']

        df_aler = df_aler[(df_aler['anomaly_score'] > 2)]

        max = df_aler['anomaly_score'].max()
        min = df_aler['anomaly_score'].min()
        df_aler['anomaly_score'] = (df_aler['anomaly_score'] - min) / (max - min)

        df_aler_ult = df_aler[:5]
        df_aler_ult = df_aler_ult[
            (df_aler_ult.index == df_aler.index.max()) | (df_aler_ult.index == ((df_aler.index.max()) - 1))
            | (df_aler_ult.index == ((df_aler.index.max()) - 2)) | (df_aler_ult.index == ((df_aler.index.max()) - 3))
            | (df_aler_ult.index == ((df_aler.index.max()) - 4))]
        if len(df_aler_ult) == 0:
            exists_anom_last_5 = 'FALSE'
        else:
            exists_anom_last_5 = 'TRUE'

        output = {'rmse': rmse, 'mse': mse, 'mae': mae, 'present_status': exists_anom_last_5,
                  'present_alerts': df_aler_ult.fillna(0).to_dict(orient='record'),
                  'past': df_aler.to_dict(orient='record'), 'model': model_type}
        # var_output['future'] = df_result_forecast.fillna(0).to_dict(orient='record')
        return output
