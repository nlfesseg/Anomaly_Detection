from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pyplot as plt
import numpy as np


# def multi_step_plot(history, true_future, prediction, scalar, step=1):
#     history = np.array(history).reshape(-1, 1)
#     history = scalar.inverse_transform(history)
#
#     true_future = np.array(true_future).reshape(-1, 1)
#     true_future = scalar.inverse_transform(true_future)
#
#     prediction = np.array(prediction).reshape(-1, 1)
#     prediction = scalar.inverse_transform(prediction)
#
#     plt.figure(figsize=(12, 6))
#     num_in = create_time_steps(len(history))
#     num_out = len(true_future)
#     plt.plot(num_in, np.array(history), label='History')
#     plt.plot(np.arange(num_out) / step, np.array(true_future), 'bo',
#              label='True Future')
#     if prediction.any():
#         plt.plot(np.arange(num_out) / step, np.array(prediction), 'ro',
#                  label='Predicted Future')
#     plt.legend(loc='upper left')
#     plt.show()

def multi_step_plot(history, actual, prediction):
    print(1)
    # x = np.arange(len(prediction) + len(history))
    # plt.plot(x[:len(history)], history, c='blue')
    # # plt.plot(x[-len(prediction):], prediction, c='red')
    # plt.plot(x[len(history):(len(history) + len(prediction))], prediction, c='red')
    # plt.plot(x[len(history):(len(history) + len(actual))], actual, c='green')
    # plt.draw()


def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    plt.show()
