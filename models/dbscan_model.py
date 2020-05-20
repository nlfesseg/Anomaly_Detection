from __future__ import absolute_import, division, print_function, unicode_literals

import os
from math import sqrt
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from kneed import KneeLocator
from models.base_model import BaseModel
from transform import replace_multiple


class DbscanModel(BaseModel):
    def __init__(self, feat_id, run_id, data=None):
        self.opt_k = 1
        self.epsilon = 0.15
        super().__init__(feat_id, run_id, data)

    def train(self, data):
        pass

    def save(self):
        joblib.dump([self.epsilon, self.opt_k],
                    os.path.join('temp', self.run_id, 'models', 'DBSCAN',
                                 '{}_DBSCAN.pkl'.format(replace_multiple(self.feat_id,
                                                                      ['/', '\\',
                                                                       ':', '?',
                                                                       '*', '"',
                                                                       '<', '>',
                                                                       '|'],
                                                                      "x"))))

    def load(self):
        self.epsilon, self.opt_k = \
            joblib.load(os.path.join('runs', self.run_id, 'models', 'DBSCAN',
                                     '{}_DBSCAN.pkl'.format(replace_multiple(self.feat_id,
                                                                          ['/', '\\',
                                                                           ':', '?',
                                                                           '*', '"',
                                                                           '<', '>',
                                                                           '|'],
                                                                          "x"))))

    def predict(self, data, scale=True):
        df_scaled = data.copy(deep=True)
        if scale:
            scalar = MinMaxScaler(feature_range=(0, 1))
            df_scaled = scalar.fit_transform(data.values)

        self.opt_k = round(sqrt(len(df_scaled)))
        nearest_neighbors = NearestNeighbors(n_neighbors=self.opt_k)
        neighbors = nearest_neighbors.fit(df_scaled)
        distances, indices = neighbors.kneighbors(df_scaled)
        distances = np.sort(distances[:, (self.opt_k - 1)], axis=0)
        i = np.arange(len(distances))
        knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')

        self.epsilon = distances[knee.knee]

        outlier_detection = DBSCAN(eps=self.epsilon, min_samples=round(sqrt(sqrt(sqrt(self.opt_k)))))
        y_pred = outlier_detection.fit_predict(df_scaled)

        n_inliers = len(y_pred) - np.count_nonzero(y_pred)
        n_outliers = np.count_nonzero(y_pred == -1)
        df = pd.DataFrame()
        df['outlier'] = y_pred.tolist()
        df['value'] = data[self.feat_id].values
        return df
