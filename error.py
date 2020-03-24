import configparser

import pandas as pd
import numpy as np


class Error:
    def __init__(self, feature, run_id):
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
        self.run_id = run_id
        # check = feature.y_val_multi[:, 0]
        self.raw_errors = list(abs(feature.y_hat - feature.y_val_multi[:, 0]))

        smoothing_window = int(self.config['LSTM_PARAMS']['BATCH_SIZE'])
        if not len(feature.y_hat) == len(feature.y_val_multi):
            raise ValueError('len(y_hat) != len(y_val_multi): {}, {}'
                             .format(len(feature.y_hat), len(feature.y_val_multi)))

        self.smoothed_errors = pd.DataFrame(self.raw_errors)\
            .ewm(span=smoothing_window).mean().values.flatten()

        self.smoothed_errors[:int(self.config['LSTM_PARAMS']['PAST_HISTORY'])] = \
            [np.mean(self.smoothed_errors[:int(self.config['LSTM_PARAMS']['PAST_HISTORY']) * 2])] \
            * int(self.config['LSTM_PARAMS']['PAST_HISTORY'])

    def detect_anomalies(self, feature):
        df = pd.DataFrame()
        df['actuals'] = feature.y_val_multi[:, 0]
        df['predicted'] = feature.y_hat
        df['error'] = self.raw_errors
        df['percentage_change'] = (self.raw_errors / feature.y_val_multi[:, 0]) * 100
        df['meanval'] = self.smoothed_errors
        df['deviation'] = pd.DataFrame(self.raw_errors).ewm(
            span=int(self.config['LSTM_PARAMS']['BATCH_SIZE'])).std().values.flatten()
        df['-3s'] = df['meanval'] - (2 * df['deviation'])
        df['3s'] = df['meanval'] + (2 * df['deviation'])
        df['-2s'] = df['meanval'] - (1.75 * df['deviation'])
        df['2s'] = df['meanval'] + (1.75 * df['deviation'])
        df['-1s'] = df['meanval'] - (1.5 * df['deviation'])
        df['1s'] = df['meanval'] + (1.5 * df['deviation'])
        cut_list = df[['error', '-3s', '-2s', '-1s', 'meanval', '1s', '2s', '3s']]
        cut_values = cut_list.values
        cut_sort = np.sort(cut_values)
        df['impact'] = [(lambda x: np.where(cut_sort == df['error'][x])[1][0])(x) for x in
                        range(len(df['error']))]
        severity = {0: 3, 1: 2, 2: 1, 3: 0, 4: 0, 5: 1, 6: 2, 7: 3}
        region = {0: "NEGATIVE", 1: "NEGATIVE", 2: "NEGATIVE", 3: "NEGATIVE", 4: "POSITIVE", 5: "POSITIVE",
                  6: "POSITIVE",
                  7: "POSITIVE"}
        df['color'] = df['impact'].map(severity)
        df['region'] = df['impact'].map(region)
        df['anomaly_points'] = np.where(df['color'] == 3, df['error'], np.nan)
        return df
