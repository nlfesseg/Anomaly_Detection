from __future__ import absolute_import, division, print_function, unicode_literals

from math import sqrt

import numpy as np
import pandas as pd
from pyod.models.knn import KNN
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i, 1)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i:i + target_size])

    return np.array(data), np.array(labels)


def replace_multiple(main_string, replaces, new_string):
    for elem in replaces:
        if elem in main_string:
            main_string = main_string.replace(elem, new_string)

    return main_string


def proces_dataset(dataset):
    dataset.drop(columns=['DateTime', 'Date Time'], inplace=True, errors='ignore')
    dif_df = dataset.diff().dropna()
    drop_cols = dif_df.columns[(dif_df == 0).sum() >= 0.98 * dif_df.shape[0]]
    dataset.drop(drop_cols, axis=1, inplace=True)
    return dataset


def adfuller_test(series, signif=0.05, name='', verbose=False):
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic': round(r[0], 4), 'pvalue': round(r[1], 4), 'n_lags': round(r[2], 4), 'n_obs': r[3]}
    p_value = output['pvalue']

    def adjust(val, length=6):
        return str(val).ljust(length)

    if p_value <= signif:
        return True
    else:
        return False


def invert_transformation(df_train, df_forecast, dif_times=1, index=-1):
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:
        # Roll back 2nd Diff
        if dif_times == 2:
            df_fc[col] = (df_train[col].iloc[index] - df_train[col].iloc[index - 1]) + df_fc[col].cumsum()
        # Roll back 1st Diff
        df_fc[col] = df_train[col].iloc[index] + df_fc[col].cumsum()
    return df_fc


def cointegration_test(df, alpha=0.05):
    out = coint_johansen(df, -1, 4)
    d = {'0.90': 0, '0.95': 1, '0.99': 2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1 - alpha)]]

    def adjust(val, length=6): return str(val).ljust(length)

    # Summary
    print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--' * 20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace, 2), 9), ">", adjust(cvt, 8), ' =>  ', trace > cvt)


def lasso(dataset, target_id):
    x = dataset.drop(target_id, axis=1)  # Feature Matrix
    y = dataset[target_id]  # Target Variable
    reg = LassoCV(cv=5)
    reg.fit(x, y)
    print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
    print("Best score using built-in LassoCV: %f" % reg.score(x, y))
    coef = pd.Series(reg.coef_, index=x.columns)

    print(
        "Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +
        str(sum(coef == 0)) + " variables")
    x = list(coef[coef != 0].index)
    x.insert(0, y.name)
    return x


def pearson(dataset, target_id):
    cor = dataset.corr()
    cor_target = abs(cor[target_id])
    correlated_features = cor_target[(cor_target > 0.5) & (cor_target < 0.99)]
    correlated_features = correlated_features.sort_values(ascending=False)
    relevant_features = [target_id]
    while not correlated_features.empty:
        x = correlated_features.iloc[1:].index  # Feature Matrix
        y = correlated_features.index[0]

        correlated_features = correlated_features.drop([y])
        relevant_features.append(y)

        for feature in x:
            y_cor = dataset[[y, feature]].corr()
            y_cor_target = abs(y_cor[y])
            y_correlated_features = y_cor_target[y_cor_target > 0.5].index[1]
            correlated_features = correlated_features.drop([y_correlated_features])
    return relevant_features


def anomaly_detection(dataset, tail, target_id, scale=True):
    if scale:
        scalar = MinMaxScaler(feature_range=(0, 1))
        scaled_dataset = scalar.fit_transform(dataset.values)
        scaled_tail = scalar.transform(tail.values)
    outliers_fraction = 0.01
    clf = KNN(contamination=outliers_fraction, n_neighbors=round(sqrt(len(dataset))))
    clf.fit(scaled_dataset)
    y_pred = clf.predict(scaled_tail)
    n_inliers = len(y_pred) - np.count_nonzero(y_pred)
    n_outliers = np.count_nonzero(y_pred == 1)
    df = pd.DataFrame()
    df['outlier'] = y_pred.tolist()
    df['value'] = tail[target_id].values
    return df


