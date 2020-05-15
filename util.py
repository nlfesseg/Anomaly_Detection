import configparser
import os
import shutil

import pandas as pd

from feature import Feature


def make_dirs(run_id):
    paths = ['temp', 'temp/%s' % run_id, 'temp/%s/models' % run_id, 'temp/%s/features' % run_id,
             'temp/%s/config' % run_id, 'temp/%s/models/VAR' % run_id, 'temp/%s/models/KNN' % run_id]
    for p in paths:
        if not os.path.isdir(p):
            os.mkdir(p)


def create_config(run_id, config_dict):
    config = configparser.ConfigParser()
    config['RUNTIME_PARAMS'] = {'VAR': 1,
                                'LSTM': 0}
    config['LSTM_PARAMS'] = {'BATCH_SIZE': 128,
                             'EPOCHS': 30,
                             'PATIENCE': 10,
                             'MIN_DELTA': 0.0003,
                             'DROPOUT': 0.03,
                             'STEP': 1}
    config['FORECAST_PARAMS'] = {'STEP_SIZE': config_dict['step_size']}
    with open(os.path.join('temp', run_id, 'config', 'config.ini'), 'w') as configfile:
        config.write(configfile)


def get_csv_filenames():
    csv_files = [file for file in os.listdir('data') if file.endswith(".csv")]
    return csv_files


def get_run_ids():
    runs = [dI for dI in os.listdir('runs') if os.path.isdir(os.path.join('runs', dI))]
    return runs


def get_feature_filenames(run_id):
    feature_filenames = [file for file in os.listdir(os.path.join('runs', run_id, 'features')) if file.endswith(".pkl")]
    return feature_filenames


def get_feature_ids(run_id):
    feature_ids = []
    for file in os.listdir(os.path.join('runs', run_id, 'features')):
        if file.endswith(".pkl"):
            feature = Feature(file, run_id)
            feature_ids.append(feature.id)
    return feature_ids


def delete_run(directory, run_id):
    shutil.rmtree(os.path.join(directory, run_id), ignore_errors=True)


def move_run(run_id):
    shutil.move(os.path.join('temp', run_id), 'runs')


def read_config(run_id):
    config = configparser.ConfigParser()
    config.read(os.path.join('runs', run_id, 'config', 'config.ini'))
    config_dict = {}
    for section in config.sections():
        config_dict[section] = {}
        for option in config.options(section):
            config_dict[section][option] = config.get(section, option)
    return config_dict


def read_dataset(filename):
    return pd.read_csv(os.path.join('data', filename))
