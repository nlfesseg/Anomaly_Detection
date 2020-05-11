from __future__ import absolute_import, division, print_function, unicode_literals

import os

import tensorflow as tf
from keras.callbacks import History, EarlyStopping

from models.base_model import BaseModel
from transform import replace_multiple, multivariate_data


class LstmModel(BaseModel):
    def __init__(self, feat_id, run_id, dataset=None):
        super().__init__(feat_id, run_id, dataset)

    def train(self, dataset):
        x_train_multi, y_train_multi = self.shape_data(dataset)
        cbs = [History(), EarlyStopping(monitor='loss',
                                        patience=int(self.config['LSTM_PARAMS']['PATIENCE']),
                                        min_delta=float(self.config['LSTM_PARAMS']['MIN_DELTA']),
                                        verbose=0)]

        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.LSTM(128,
                                            activation='relu',
                                            input_shape=(x_train_multi.shape[1], x_train_multi.shape[2])))
        self.model.add(tf.keras.layers.Dropout(float(self.config['LSTM_PARAMS']['DROPOUT'])))
        self.model.add(tf.keras.layers.RepeatVector(int(self.config['FORECAST_PARAMS']['FUTURE_TARGET']) + int(self.config['FORECAST_PARAMS']['PAST_HISTORY'])))
        self.model.add(tf.keras.layers.LSTM(128, return_sequences=True, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(float(self.config['LSTM_PARAMS']['DROPOUT'])))
        self.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(x_train_multi.shape[2])))

        self.model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        self.model.fit(x_train_multi, y_train_multi,
                       batch_size=int(self.config['LSTM_PARAMS']['BATCH_SIZE']),
                       epochs=int(self.config['LSTM_PARAMS']['EPOCHS']),
                       callbacks=cbs)

    def save(self):
        self.model.save(os.path.join('temp', self.run_id, 'models', 'LSTM',
                                     '{}_LSTM.h5'.format(replace_multiple(self.feat_id,
                                                                          ['/', '\\', ':', '?', '*', '"', '<', '>',
                                                                           '|'],
                                                                          "x"))))

    def load(self):
        self.model = tf.keras.models.load_model(os.path.join('runs', self.run_id,
                                                             'models', 'LSTM',
                                                             '{}_LSTM.h5'.format(replace_multiple(self.feat_id,
                                                                                                  ['/', '\\', ':', '?',
                                                                                                   '*', '"', '<', '>',
                                                                                                   '|'],
                                                                                                  "x"))))

    def predict(self, dataset):
        start_idx = ((int(self.config['FORECAST_PARAMS']['PAST_HISTORY']) + int(
            self.config['FORECAST_PARAMS']['FUTURE_TARGET'])) * 4) + int(
            self.config['FORECAST_PARAMS']['PAST_HISTORY'])
        end_idx = int(self.config['FORECAST_PARAMS']['PAST_HISTORY'])

        n_input = dataset[-start_idx:-end_idx].values
        n_input = n_input.reshape(1, n_input.shape[0], n_input.shape[1])
        prediction = self.model.predict(n_input)
        return prediction

    def shape_data(self, dataset, end_index=None):
        # column_loc = dataset.columns.get_loc(self.feat_id)
        dataset = dataset.values
        history_size = (int(self.config['FORECAST_PARAMS']['PAST_HISTORY']) + int(self.config['FORECAST_PARAMS']['FUTURE_TARGET'])) * 4
        target_size = int(self.config['FORECAST_PARAMS']['PAST_HISTORY']) + int(self.config['FORECAST_PARAMS']['FUTURE_TARGET'])
        x_train_multi, y_train_multi = multivariate_data(dataset, dataset, 0,
                                                         end_index,
                                                         history_size,
                                                         target_size)
        return x_train_multi, y_train_multi

    def result(self, history, actual, prediction, forecast, df_aler, model_type):
        pass
