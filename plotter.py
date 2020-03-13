from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import configparser
import plotly.graph_objects as go
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot
from plotly.offline import iplot

from util import create_time_steps


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

def multi_step_plot(actual, prediction, scalar):
    actual = np.array(actual).reshape(-1, 1)
    actual = scalar.inverse_transform(actual)

    prediction = np.array(prediction).reshape(-1, 1)
    prediction = scalar.inverse_transform(prediction)
    plt.plot(actual)
    plt.plot(prediction)
    plt.show()


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


def plot_anomaly(df, metric_name):
    # error = pd.DataFrame(Order_results.error.values)
    # df = df.sort_values(by='load_date', ascending=False)
    # df.load_date = pd.to_datetime(df['load_date'].astype(str), format="%Y%m%d")
    # meanval = error.rolling(window=window).mean()
    # deviation = error.rolling(window=window).std()
    # res = error
    # upper_bond=meanval + (2 * deviation)
    # lower_bond=meanval - (2 * deviation)
    # anomalies = pd.DataFrame(index=res.index, columns=res.columns)
    # anomalies[res < lower_bond] = res[res < lower_bond]
    # anomalies[res > upper_bond] = res[res > upper_bond]
    bool_array = (abs(df['anomaly_points']) > 0)
    # And a subplot of the Actual Values.
    actuals = df["actuals"][-len(bool_array):]
    anomaly_points = bool_array * actuals
    anomaly_points[anomaly_points == 0] = np.nan
    # Order_results['meanval']=meanval
    # Order_results['deviation']=deviation
    color_map = {0: "'rgba(228, 222, 249, 0.65)'", 1: "yellow", 2: "orange", 3: "red"}
    table = go.Table(
        domain=dict(x=[0, 1],
                    y=[0, 0.3]),
        columnwidth=[1, 2],
        # columnorder=[0, 1, 2,],
        header=dict(height=20,
                    values=[['<b>Actual Values </b>'],
                            ['<b>Predicted</b>'], ['<b>% Difference</b>'], ['<b>Severity (0-3)</b>']],
                    font=dict(color=['rgb(45, 45, 45)'] * 5, size=14),
                    fill=dict(color='#d562be')),
        cells=dict(values=[df.round(3)[k].tolist() for k in ['actuals', 'predicted',
                                                             'percentage_change', 'color']],
                   line=dict(color='#506784'),
                   align=['center'] * 5,
                   font=dict(color=['rgb(40, 40, 40)'] * 5, size=12),
                   # format = [None] + [",.4f"] + [',.4f'],
                   # suffix=[None] * 4,
                   suffix=[None] + [''] + [''] + ['%'] + [''],
                   height=27,
                   # fill = dict(color=['rgb(235, 193, 238)', 'rgba(228, 222, 249, 0.65)']))
                   fill=dict(color=[df['color'].map(color_map)],
                             )
                   ))
    # df['ano'] = np.where(df['color']==3, df['error'], np.nan)
    anomalies = go.Scatter(name="Anomaly",
                           xaxis='x1',
                           yaxis='y1',
                           y=df['anomaly_points'],
                           mode='markers',
                           marker=dict(color='red',
                                       size=11, line=dict(
                                   color="red",
                                   width=2)))
    upper_bound = go.Scatter(hoverinfo="skip",
                             showlegend=False,
                             xaxis='x1',
                             yaxis='y1',
                             y=df['3s'],
                             marker=dict(color="#444"),
                             line=dict(
                                 color=('rgb(23, 96, 167)'),
                                 width=2,
                                 dash='dash'),
                             fillcolor='rgba(68, 68, 68, 0.3)',
                             fill='tonexty')
    lower_bound = go.Scatter(name='Confidence Interval',
                             xaxis='x1',
                             yaxis='y1',
                             y=df['-3s'],
                             marker=dict(color="#444"),
                             line=dict(
                                 color=('rgb(23, 96, 167)'),
                                 width=2,
                                 dash='dash'),
                             fillcolor='rgba(68, 68, 68, 0.3)',
                             fill='tonexty')
    Actuals = go.Scatter(name='Actuals',

                         y=df['actuals'],
                         xaxis='x2', yaxis='y2',
                         mode='line',
                         marker=dict(size=12,
                                     line=dict(width=1),
                                     color="blue"))
    Predicted = go.Scatter(name='Predicted',
                           y=df['predicted'],
                           xaxis='x2', yaxis='y2',
                           mode='line',
                           marker=dict(size=12,
                                       line=dict(width=1),
                                       color="orange"))
    Error = go.Scatter(name="Error",
                       y=df['error'],
                       xaxis='x1',
                       yaxis='y1',
                       mode='line',
                       marker=dict(size=12,
                                   line=dict(width=1),
                                   color="red"),
                       text="Error")
    anomalies_map = go.Scatter(name="anomaly actual",
                               showlegend=False,
                               y=anomaly_points,
                               mode='markers',
                               xaxis='x2',
                               yaxis='y2',
                               marker=dict(color="red",
                                           size=11,
                                           line=dict(
                                               color="red",
                                               width=2)))
    Mvingavrg = go.Scatter(name="Moving Average",
                           y=df['meanval'],
                           mode='line',
                           xaxis='x1',
                           yaxis='y1',
                           marker=dict(size=12,
                                       line=dict(width=1),
                                       color="green"),
                           text="Moving average")
    axis = dict(
        showline=True,
        zeroline=False,
        showgrid=True,
        mirror=True,
        ticklen=4,
        gridcolor='#ffffff',
        tickfont=dict(size=10))
    layout = dict(
        width=1000,
        height=865,
        autosize=False,
        title=metric_name,
        margin=dict(t=75),
        showlegend=True,
        xaxis1=dict(axis, **dict(domain=[0, 1], anchor='y1', showticklabels=True)),
        xaxis2=dict(axis, **dict(domain=[0, 1], anchor='y2', showticklabels=True)),
        yaxis1=dict(axis, **dict(domain=[2 * 0.21 + 0.20 + 0.09, 1], anchor='x1', hoverformat='.2f')),
        yaxis2=dict(axis, **dict(domain=[0.21 + 0.12, 2 * 0.31 + 0.02], anchor='x2', hoverformat='.2f')))

    fig = go.Figure(data=[table, anomalies, anomalies_map,
                          upper_bound, lower_bound, Actuals, Predicted,
                          Mvingavrg, Error], layout=layout)
    iplot(fig)
    pyplot.show()
