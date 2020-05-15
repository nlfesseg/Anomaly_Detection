from __future__ import absolute_import, division, print_function, unicode_literals

import configparser
import os
from datetime import datetime as dt

from feature import Feature
from transform import proces_dataset
from util import make_dirs, create_config, read_dataset, get_feature_filenames, delete_run, move_run


class Detector:
    def __init__(self, filename):
        self.training = False
        self.sized_dataset = None
        self.selected_feature_ids = None
        self.features = []
        self.filename = filename

        self.config = configparser.ConfigParser()
        self.dataset = read_dataset(self.filename)
        self.dataset = proces_dataset(self.dataset)

    def load(self, run_id):
        self.config.read(os.path.join('runs', run_id, 'config', 'config.ini'))
        self.sized_dataset = self.dataset.iloc[::int(self.config['FORECAST_PARAMS']['STEP_SIZE']), :]

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
        self.sized_dataset = self.dataset.iloc[::int(self.config['FORECAST_PARAMS']['STEP_SIZE']), :]
        self.sized_dataset = proces_dataset(self.sized_dataset)

        self.features = []
        for i in range(self.sized_dataset.shape[1]):
            if self.training:
                feature = Feature(self.sized_dataset.columns[i], run_id, self.sized_dataset)
                feature.train_models(self.sized_dataset)
                self.features.append(feature)
            else:
                delete_run('temp', run_id)
                return False
        move_run(run_id)
        return True

    def predict(self, view_history, past_history, future_target):
        results = []
        for feature_id in self.selected_feature_ids:
            feature = next((x for x in self.features if x.id == feature_id), None)
            results.append(feature.predict(self.sized_dataset, view_history, past_history, future_target))
        return results

    def update(self):
        modified_dataset = read_dataset(self.filename)
        if (len(modified_dataset) % int(self.config['FORECAST_PARAMS']['STEP_SIZE'])) == 0:
            self.dataset = modified_dataset[self.dataset.columns.values]
            self.sized_dataset = self.dataset.iloc[::int(self.config['FORECAST_PARAMS']['STEP_SIZE']), :]
            # self.scaled_data = self.sized_dataset.copy(deep=True)
            # for feature in self.features:
            #     self.scaled_data[feature.id] = feature.scalar.transform(
            #         self.scaled_data[feature.id].values.reshape(-1, 1))
            return True
        return False

    def is_valid(self):
        if len(self.dataset.columns) >= 1:
            if len(self.dataset >= 1):
                return True
        return False
