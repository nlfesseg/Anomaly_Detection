from __future__ import absolute_import, division, print_function, unicode_literals

import os
from math import sqrt
import numpy as np
import joblib
import pandas as pd
from pyod.models.knn import KNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from models.base_model import BaseModel
from transform import replace_multiple


class KnnModel(BaseModel):
    def __init__(self, feat_id, run_id, dataset=None):
        self.optimal_k = 1
        super().__init__(feat_id, run_id, dataset)

    def train(self, dataset):
        self.optimal_k = round(sqrt(len(dataset)))

    def save(self):
        joblib.dump(self.optimal_k,
                    os.path.join('temp', self.run_id, 'models', 'KNN',
                                 '{}_KNN.pkl'.format(replace_multiple(self.feat_id,
                                                                      ['/', '\\',
                                                                       ':', '?',
                                                                       '*', '"',
                                                                       '<', '>',
                                                                       '|'],
                                                                      "x"))))

    def load(self):
        self.optimal_k = \
            joblib.load(os.path.join('runs', self.run_id, 'models', 'KNN',
                                     '{}_KNN.pkl'.format(replace_multiple(self.feat_id,
                                                                          ['/', '\\',
                                                                           ':', '?',
                                                                           '*', '"',
                                                                           '<', '>',
                                                                           '|'],
                                                                          "x"))))

    def predict(self, train_data, test_data, scale=True):
        scaled_test_data = test_data.copy(deep=True)
        if scale:
            scalar = MinMaxScaler(feature_range=(0, 1))
            train_data = scalar.fit_transform(train_data.values)
            scaled_test_data = scalar.transform(scaled_test_data.values)
        # outliers_fraction = 0.01
        # clf = KNN(contamination=outliers_fraction, n_neighbors=self.optimal_k)
        # clf.fit(train_data)
        # y_pred = clf.predict(scaled_test_data)

        outlier_detection = DBSCAN(eps=.2, min_samples=5, n_jobs=-1)
        outlier_detection.fit(scaled_test_data)
        # y_pred = outlier_detection.predict(scaled_test_data)

        n_inliers = len(outlier_detection.labels_) - np.count_nonzero(outlier_detection.labels_)
        n_outliers = np.count_nonzero(outlier_detection.labels_ == -1)
        df = pd.DataFrame()
        df['outlier'] = outlier_detection.labels_.tolist()
        df['value'] = test_data[self.feat_id].values
        return df
