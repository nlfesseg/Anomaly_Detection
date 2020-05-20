from __future__ import absolute_import, division, print_function, unicode_literals

import configparser
import os
from datetime import datetime as dt

from feature import Feature
from transform import proces_data
from util import make_dirs, create_config, get_feature_filenames, delete_run, move_run, read_data


class Detector:
    def __init__(self, filename):
        self.training = False
        self.df_sized = None
        self.selected_feature_ids = None
        self.features = []
        self.filename = filename

        self.config = configparser.ConfigParser()
        self.df = read_data(self.filename)
        self.df = proces_data(self.df)

    def load(self, run_id):
        self.config.read(os.path.join('runs', run_id, 'config', 'config.ini'))
        self.df_sized = self.df.iloc[::int(self.config['FORECAST_PARAMS']['STEP_SIZE']), :]

        self.features = []
        filenames = get_feature_filenames(run_id)
        for file in filenames:
            feature = Feature(file, run_id)
            feature.load_models()
            self.features.append(feature)

    def train(self, config_dict):
        run_id = dt.now().strftime('%Y-%m-%d_%H.%M.%S')
        make_dirs(run_id)
        create_config(run_id, config_dict)

        self.config.read(os.path.join('temp', run_id, 'config', 'config.ini'))
        self.df_sized = self.df.iloc[::int(self.config['FORECAST_PARAMS']['STEP_SIZE']), :]
        self.df_sized = proces_data(self.df_sized)

        self.features = []
        for i in range(self.df_sized.shape[1]):
            if self.training:
                feature = Feature(self.df_sized.columns[i], run_id, self.df_sized)
                feature.train_models(self.df_sized)
                self.features.append(feature)
            else:
                delete_run('temp', run_id)
                return False
        move_run(run_id)
        return True

    def predict(self, past_history, future_target, view_history, cluster_history):
        if view_history < past_history:
            view_history = past_history
        if cluster_history < view_history:
            cluster_history = view_history

        if past_history > (len(self.df_sized) * 0.1):
            past_history = (len(self.df_sized) * 0.1)
        if future_target > (len(self.df_sized) * 0.1):
            future_target = (len(self.df_sized) * 0.1)

        if view_history > len(self.df_sized):
            view_history = len(self.df_sized)
        if cluster_history > len(self.df_sized):
            cluster_history = len(self.df_sized)

        results = []
        for feature_id in self.selected_feature_ids:
            feature = next((x for x in self.features if x.id == feature_id), None)
            results.append(
                feature.predict(self.df_sized, past_history, future_target, view_history, cluster_history))
        return results

    def update(self):
        df_mod = read_data(self.filename)
        if (len(df_mod) % int(self.config['FORECAST_PARAMS']['STEP_SIZE'])) == 0:
            self.df = df_mod[self.df.columns.values]
            self.df_sized = self.df.iloc[::int(self.config['FORECAST_PARAMS']['STEP_SIZE']), :]
            return True
        return False
