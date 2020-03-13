from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from keras.callbacks import History, EarlyStopping

import plotter as plt
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

        plt.plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')

    def save(self):
        self.model.save(os.path.join('data', self.run_id, 'models',
                                     '{}_LSTM.h5'.format(replace_multiple(self.feat_id,
                                                                          ['/', '\\', ':', '?', '*', '"', '<', '>',
                                                                           '|'],
                                                                          "x"))))

    def load(self):
        self.model = tf.keras.models.load_model(os.path.join('data', self.config['RUNTIME_PARAMS']['USE_ID'],
                                                             'models',
                                                             '{}_LSTM.h5'.format(replace_multiple(self.feat_id,
                                                                                                  ['/', '\\', ':', '?',
                                                                                                   '*', '"', '<', '>',
                                                                                                   '|'],
                                                                                                  "x"))))

    def predict(self, feature):
        predictions = []
        for i in range(int(len(feature.x_val_multi) / 10)):
            index = i * 10
            n_input = feature.x_val_multi[index].reshape(1, feature.x_val_multi[index].shape[0],
                                                         feature.x_val_multi[index].shape[1])
            prediction = self.model.predict(n_input)[0]
            predictions.append(prediction)

        predictions = np.concatenate(np.array(predictions))
        actual = np.reshape(feature.y_val_multi[0: -4, 0])
        plt.multi_step_plot(actual, predictions, feature.scalar)
        # plt.multi_step_plot(feature.x_val_multi[index][:, -1], feature.y_val_multi[index],
        #                     self.model.predict(n_input)[0],
        #                     feature.scalar)

    def aggregate_predictions(self, y_pred_batch, method='first'):
        agg_y_pred_batch = np.array([])

        for t in range(len(y_pred_batch)):

            start_idx = t - int(self.config['LSTM_PARAMS']['FUTURE_TARGET'])
            start_idx = start_idx if start_idx >= 0 else 0

            y_pred_t = np.flipud(y_pred_batch[start_idx:t + 1]).diagonal()

            if method == 'first':
                agg_y_pred_batch = np.append(agg_y_pred_batch, [y_pred_t[0]])
            elif method == 'mean':
                agg_y_pred_batch = np.append(agg_y_pred_batch, np.mean(y_pred_t))

        agg_y_pred_batch = agg_y_pred_batch.reshape(len(agg_y_pred_batch), 1)
        self.y_pred = np.append(self.y_pred, agg_y_pred_batch)

    def batch_predict(self, feature):
        feature.x_val_multi = np.concatenate((feature.x_val_multi, feature.x_val_multi_split), axis=0)
        temp_array = np.repeat(feature.y_val_multi[-1], int(self.config['LSTM_PARAMS']['FUTURE_TARGET']))\
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
            y_pred_batch = self.model.predict(x_val_batch)
            self.aggregate_predictions(y_pred_batch)

        self.y_pred = np.reshape(self.y_pred, (self.y_pred.size,))

        feature.y_pred = self.y_pred
        return feature

    def result(self, feature, model):
        pass
