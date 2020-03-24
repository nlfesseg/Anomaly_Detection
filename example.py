from __future__ import absolute_import, division, print_function, unicode_literals

import configparser
from datetime import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from feature import Feature
from models.lstm_model import LstmModel
from util import make_dirs, adfuller_test
from models.var_model import VarModel

df = pd.read_csv("out.csv")
df = df.drop(columns='DateTime')
drop_cols = df.columns[(df == 0).sum() > 0.9 * df.shape[0]]
df.drop(drop_cols, axis=1, inplace=True)
dataset = df.loc[:, (df != df.iloc[0]).any()]

config = configparser.ConfigParser()
config.read('config.ini')
scalars = []

id = dt.now().strftime('%Y-%m-%d_%H.%M.%S')
make_dirs(id)


def run():
    features = []
    for i in range(dataset.shape[1]):
        feature = Feature(dataset.columns[i], id, dataset)
        dataset.iloc[:, i] = feature.scalar.transform(dataset.iloc[:, i].values.reshape(-1, 1))
        features.append(feature)

    for i in range(len(features)):
        feature = features[i]
        features_considered_values = dataset[feature.features_considered_ids]
        feature.shape_data(features_considered_values)
        if config['RUNTIME_PARAMS'].getboolean('PREDICT'):
            try:
                pass
                lstm_model = LstmModel(feature, id)
                # feature_lstm = lstm_model.batch_predict(feature)
                lstm_model.predict(feature)
                # result = super(LstmModel, lstm_model).result(feature_lstm, 'LSTM')
            except Exception as e:
                print(e)
                print('ERROR: exception executing LSTM')
            # try:
            #     pass
            #     var_model = VarModel(feature, id)
            #     feature_var = var_model.predict(feature)
            #     # result = super(VarModel, var_model).result(feature_var, 'VAR')
            #     # print(result)
            # except Exception as e:
            #     print(e)
            #     print('ERROR: exception executing VAR')

        # predicted_df = pd.DataFrame()
        # predicted_df['actuals'] = feature.y_val_multi[:, 0]
        # predicted_df['predicted'] = feature.y_pred
        # predicted_df.reset_index(inplace=True)


run()
