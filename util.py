from __future__ import absolute_import, division, print_function, unicode_literals

import configparser
import os

import numpy as np


def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i:i + target_size])

    return np.array(data), np.array(labels)


def make_dirs(_id):
    config = configparser.ConfigParser()
    config.read('config.ini')

    if not config['RUNTIME_PARAMS'].getboolean('TRAIN') or not config['RUNTIME_PARAMS'].getboolean('PREDICT'):
        if not os.path.isdir('data/%s' % config['RUNTIME_PARAMS']['USE_ID']):
            raise ValueError(
                "Run ID {} is not valid. If loading prior models or predictions, must provide valid ID.".format(_id))

    paths = ['data', 'data/%s' % _id, 'data/%s/models' % _id, 'data/%s/scalars' % _id]

    for p in paths:
        if not os.path.isdir(p):
            os.mkdir(p)


def create_time_steps(length):
    return list(range(-length, 0))


def replace_multiple(main_string, replaces, new_string):
    # Iterate over the strings to be replaced
    for elem in replaces:
        # Check if string is in the main string
        if elem in main_string:
            # Replace the string
            main_string = main_string.replace(elem, new_string)

    return main_string
