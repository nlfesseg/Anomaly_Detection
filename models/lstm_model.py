from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from keras.callbacks import History, EarlyStopping
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from models.base_model import BaseModel
from util import replace_multiple, multivariate_data


class LstmModel(BaseModel):
    def __init__(self, feature, run_id):
        super().__init__(feature, run_id)

    def train(self, feature):
        cbs = [History(), EarlyStopping(monitor='val_loss',
                                        patience=int(self.config['LSTM_PARAMS']['PATIENCE']),
                                        min_delta=float(self.config['LSTM_PARAMS']['MIN_DELTA']),
                                        verbose=0)]

        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.LSTM(128,
                                            return_sequences=True,
                                            input_shape=(None, feature.x_train_multi.shape[2])))
        self.model.add(tf.keras.layers.Dropout(float(self.config['LSTM_PARAMS']['DROPOUT'])))

        self.model.add(tf.keras.layers.LSTM(128, return_sequences=False, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(float(self.config['LSTM_PARAMS']['DROPOUT'])))
        self.model.add(tf.keras.layers.Dense(int(self.config['LSTM_PARAMS']['FUTURE_TARGET'])))

        self.model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        multi_step_history = self.model.fit(feature.x_train_multi, feature.y_train_multi,
                                            batch_size=int(self.config['LSTM_PARAMS']['BATCH_SIZE']),
                                            epochs=int(self.config['LSTM_PARAMS']['EPOCHS']),
                                            callbacks=cbs,
                                            validation_data=(feature.x_val_multi, feature.y_val_multi))

        # plt.plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')

    def save(self):
        self.model.save(os.path.join('data', self.run_id, 'models', 'LSTM',
                                     '{}_LSTM.h5'.format(replace_multiple(self.feat_id,
                                                                          ['/', '\\', ':', '?', '*', '"', '<', '>',
                                                                           '|'],
                                                                          "x"))))

    def load(self):
        self.model = tf.keras.models.load_model(os.path.join('data', self.config['RUNTIME_PARAMS']['USE_ID'],
                                                             'models', 'LSTM',
                                                             '{}_LSTM.h5'.format(replace_multiple(self.feat_id,
                                                                                                  ['/', '\\', ':', '?',
                                                                                                   '*', '"', '<', '>',
                                                                                                   '|'],
                                                                                                  "x"))))

    def predict(self, feature):
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(1, len(feature.x_val_multi)):
            n_input = feature.x_val_multi[i].reshape(1, feature.x_val_multi[i].shape[0],
                                                     feature.x_val_multi[i].shape[1])
            forecast = self.model.predict(n_input).reshape(-1, 1)
            forecast = feature.scalar.inverse_transform(forecast)

            history = np.array(feature.x_val_multi[i]).reshape(-1, 1)
            history = feature.scalar.inverse_transform(history)

            actual = feature.y_val_multi[i].reshape(-1, 1)
            actual = feature.scalar.inverse_transform(actual)
            # plt.multi_step_plot(history, actual, forecast)
            x = np.arange(len(forecast) + len(history))
            ax.plot(x[:len(history)], history, c='blue')
            # plt.plot(x[-len(prediction):], prediction, c='red')
            ax.plot(x[len(history):(len(history) + len(forecast))], forecast, c='red')
            ax.plot(x[len(history):(len(history) + len(actual))], actual, c='green')
            plt.show()
            plt.close(fig)

    def aggregate_predictions(self, y_hat_batch, method='first'):
        agg_y_hat_batch = np.array([])

        for t in range(len(y_hat_batch)):

            start_idx = t - int(self.config['LSTM_PARAMS']['FUTURE_TARGET'])
            start_idx = start_idx if start_idx >= 0 else 0

            y_hat_t = np.flipud(y_hat_batch[start_idx:t + 1]).diagonal()

            if method == 'first':
                agg_y_hat_batch = np.append(agg_y_hat_batch, [y_hat_t[0]])
            elif method == 'mean':
                agg_y_hat_batch = np.append(agg_y_hat_batch, np.mean(y_hat_t))

        agg_y_hat_batch = agg_y_hat_batch.reshape(len(agg_y_hat_batch), 1)
        self.y_hat = np.append(self.y_hat, agg_y_hat_batch)

    def batch_predict(self, feature):
        feature.x_val_multi = np.concatenate((feature.x_val_multi, feature.x_val_multi_split), axis=0)
        temp_array = np.repeat(feature.y_val_multi[-1], int(self.config['LSTM_PARAMS']['FUTURE_TARGET'])) \
            .reshape(-1, int(self.config['LSTM_PARAMS']['FUTURE_TARGET']))
        feature.y_val_multi = np.concatenate((feature.y_val_multi, temp_array), axis=0)
        num_batches = int((feature.y_val_multi.shape[0] - int(self.config['LSTM_PARAMS']['PAST_HISTORY']))
                          / int(self.config['LSTM_PARAMS']['BATCH_SIZE']))
        if num_batches < 0:
            raise ValueError('Number of batches is 0.')

        for i in range(0, num_batches + 1):
            prior_idx = i * int(self.config['LSTM_PARAMS']['BATCH_SIZE'])
            idx = (i + 1) * int(self.config['LSTM_PARAMS']['BATCH_SIZE'])

            if i + 1 == num_batches + 1:
                idx = feature.y_val_multi.shape[0]

            x_val_batch = feature.x_val_multi[prior_idx:idx]
            y_hat_batch = self.model.predict(x_val_batch)
            self.aggregate_predictions(y_hat_batch)

        # last_observation = feature.x_val_multi[-30]
        # forecast = self.model.predict(last_observation.reshape(1, last_observation.shape[0], last_observation.shape[1]))
        # self.y_hat = np.concatenate((self.y_hat, forecast.flatten()), axis=0)
        #
        # self.y_hat = pd.DataFrame(data=self.y_hat,
        #                           columns=[self.feat_id])

        feature.y_hat = self.y_hat
        return feature

    def result(self, feature, model):
        pass
