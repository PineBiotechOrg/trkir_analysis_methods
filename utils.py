import os
import datetime
import math
import sys
import csv
import random
from collections import Counter

import pandas as pd
import numpy as np
import itertools

from keras.models import Sequential
from scipy.stats import pearsonr
from scipy.cluster.hierarchy import dendrogram, linkage
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw, dtw
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from hmmlearn import hmm
import joblib

from config import *
from ssa_class import SSA


def load_csv(csv_name, cols):
    with open(csv_name, newline='') as csvfile:
        return pd.read_csv(csvfile, usecols=cols)


def clear_from_nans(data):
    for key in data.keys():
        data[key] = np.nan_to_num(data[key])

    return data


def pd_to_csv(pd, csv_name):
    pd.to_csv(csv_name)


def save_csv(data, csv_name):
    with open(csv_name, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(data)


def divide_by_days(data, full_days=False):
    day_indexes = [0]

    dates = data[FEATURES[0]]

    prev_day = dates[0].split('/')[1]
    for i in range(len(dates)):
        if dates[i].split('/')[1] != prev_day:
            prev_day = dates[i].split('/')[1]
            day_indexes.append(i)

    day_indexes.append(len(dates) - 1)

    if full_days:
        # del day_indexes[0]
        del day_indexes[-1]
    return {
        key: [list(data[key][day_indexes[i]: day_indexes[i + 1]]) for i in range(len(day_indexes) - 1)]
        for key in FEATURES
    }


def add_mock_last_day(data):
    day_len = len(data[FEATURES[0]][len(data[FEATURES[0]]) - 1])
    for key in data.keys():
        del data[key][0]
        data[key][-1] = data[key][-1] + [data[key][-1][-1]] * (day_len - len(data[key][-1]))

    return data


def delete_bad_days(data):
    for i, day in enumerate(data['temp']):
        if np.mean(day) < -1:
            break
        bad_day_index = i

    for key in data.keys():
        data[key] = data[key][:bad_day_index + 1]

    return data


def divide_days_by_time(data, parts):
    splitted_by_time = {key: [] for key in FEATURES}

    for key in FEATURES:
        for day in range(len(data[key])):
            splitted_by_time[key].append(np.array(np.array_split(data[key][day], parts)).tolist())

    return splitted_by_time


def average_by_table(data):
    averaged = {
        key: np.mean(np.array(data[key]).astype(float)) for key in FEATURES[2:]
    }
    averaged[FEATURES[0]] = data[FEATURES[0]][0]
    averaged[FEATURES[1]] = data[FEATURES[1]][0]

    return averaged


def average_by_days(data):
    averaged = {
        key: [np.mean(np.array(data[key][day]).astype(float)) for day in range(len(data[FEATURES[0]]))]
        for key in FEATURES[2:]
    }
    averaged[FEATURES[0]] = [data[FEATURES[0]][day][0] for day in range(len(data[FEATURES[0]]))]
    averaged[FEATURES[1]] = [data[FEATURES[1]][day][0] for day in range(len(data[FEATURES[1]]))]

    return averaged


def average_csv_days(data):
    averaged = {
        key: np.mean(np.array(data[key]).astype(float)) for key in FEATURES[2:]
    }

    if FEATURES[0] in data.keys():
        averaged[FEATURES[0]] = data[FEATURES[0]][0][0]
        averaged[FEATURES[1]] = data[FEATURES[1]][0][0]

    return averaged


def average_days_by_time(data):
    # print(data['time'][0][-1])
    averaged = {
        key: [[np.mean(np.array(data[key][day][part]).astype(float)) for part in range(len(data[key][day]))]
              for day in range(len(data[FEATURES[0]]))]
        for key in FEATURES[2:]
    }
    averaged[FEATURES[0]] = [[data[FEATURES[0]][day][part][0] for part in range(len(data[FEATURES[0]][day]))]
                             for day in range(len(data[FEATURES[0]]))]
    averaged[FEATURES[1]] = [[data[FEATURES[1]][day][part][0] for part in range(len(data[FEATURES[1]][day]))]
                             for day in range(len(data[FEATURES[1]]))]

    return averaged


def compose_night(night_indexes):
    #    night = []
    #
    #    a = np.array(night_indexes).reshape((-1, 24))
    #    first_night = set(a[0])
    #
    #
    # #   print(night_indexes)
    #    for i in range(len(night_indexes)):
    #        if night_indexes[i] == 'night':
    #            night.append(i % 24)
    #
    #    night = list(set(night))
    #  #  print(night)

    return ['night' if i >= 15 else 'day' for i in range(24)]


def average_days_by_activity(data, activity='day'):
    nights = compose_night([])
    avg_data = {
        key: [np.mean(np.array(data[key][day])[np.where(np.array(nights) == activity)].tolist()) for day in
              range(len(data[key]))]
        for key in FEATURES[2:]
    }

    return avg_data


def plot_one_subplot(datas, titles, save=False, save_name=None):
    plt.figure(figsize=(14, 10))

    for i, data in enumerate(datas):
        plt.plot(np.arange(len(data)), np.array(data) - 0.6 * (i + 1), label=titles[i])

    plt.legend()

    if save:
        plt.savefig('search/' + save_name + '.png')
    else:
        plt.show()


def plot_feature(datas, feature, titles=[], save=False, x=None):
    fig = plt.figure(figsize=(14, 10))

    for i, data in enumerate(datas):
        ax = fig.add_subplot(len(datas), 1, i + 1)
        ax.set_xlabel('x', fontsize=5)
        ax.set_ylabel('y', fontsize=5)
        ax.set_title(feature if not titles else titles[i], fontsize=10)

        if x is not None:
            ax.plot(x[i], data[feature])
        else:
            ax.plot(data[feature])

    if save:
        plt.savefig(os.path.join('plots', 'tw', titles[-1].replace('/', '_') + '.png'))
        plt.close()
    else:
        plt.show()


def moving_average(data, window_size=100, step=1):
    ma_data = []
    for i in range(0, len(data), step):
        part = float(i) / (len(data) - 1)
        to_left = int(window_size * part)
        to_right = window_size - to_left

        ma_data.append(np.mean(data[i - to_left: i + to_right]))

    return ma_data


def trendline(data, deg=1):
    coeffs = np.polyfit(np.arange(len(data)), data, deg)
    trend = np.poly1d(coeffs)

    return trend(np.arange(len(data)))


def plot_feature_with_trends(data, dates=None, title='temp', trend='ma', colors=None, save=False):
    plt.figure(figsize=(14, 10))

    plt.title(title, fontsize=15)

    if dates:
        dates_ticks = [i for i in range(0, len(dates), 144)]
        plt.xticks(dates_ticks, [dates[int(i)] for i in dates_ticks], rotation=10)

    for i in range(len(data)):
        #    csv_name = titles[i]
        #     color = 'blue'
        # if 'april_healthy_4_pc' in csv_name or 'june_healthy_1_ucsf' in csv_name or 'june_healthy_2_ucsf' in csv_name or 'june_healthy_2_pc' in csv_name or 'june_healthy_3_pc' in csv_name:
        #     color = 'green'

        if colors:
            plt.plot(np.arange(len(data[i])), data[i], color=colors[i], alpha=0.7, linewidth=1)
        else:
            plt.plot(np.arange(len(data[i])), data[i], alpha=0.7, linewidth=1)

        color = 'green'
        if trend == 'ma':
            ma = moving_average(data[i], window_size=100)
            plt.plot(np.arange(len(ma)), ma, color, linewidth=5)
        if trend == 'tl':
            plt.plot(trendline(data[i]))

    if save:
        plt.savefig(os.path.join('plots', trend, title.replace('/', '_') + '.png'))
        plt.close()
    else:
        plt.show()


def kstest(data, healthy_days, feature, csv_name, csv_opposite_name, period_length=8):
    healthy_day = healthy_days[feature]

    ks_scores = []
    for day_num, day in enumerate(data[feature]):
        parts_count = len(day) // period_length
        for i in range(parts_count - 1):
            healthy_part = healthy_day[i * period_length: (i + 1) * period_length]
            current_part = day[i * period_length:(i + 1) * period_length]
            ks_scores.append(list(ks_2samp(healthy_part, current_part))[-1])
            ks_scores[-1] = 0 if ks_scores[-1] < KS_THRESH else 1

    return pd.DataFrame(data=ks_scores, columns=['ks_scores'])


def fft(data, rate):
    spec_x = np.fft.fftfreq(len(data), d=1.0 / rate)
    spec_x = spec_x[3: len(spec_x) // 2]
    y = np.fft.fft(data)
    spec_y = np.abs(y)[3: len(y) // 2]

    return spec_x, spec_y


def random_forest(data):
    rf = RandomForestClassifier()
    rf.fit(data[FEATURES[2:]], data['target'])

    return pd.DataFrame(rf.feature_importances_,
                        index=FEATURES[2:],
                        columns=['importance']).sort_values('importance', ascending=False)


def regression(data, average, days=[0], transform=True):
    to_compare = np.array(list(itertools.chain.from_iterable([data[day] for day in days]))).reshape(-1, 1)
    with_compare = np.array(list(itertools.chain.from_iterable([average[day] for day in days]))).reshape(-1, 1)

    reg = LinearRegression(copy_X=True).fit(to_compare, with_compare)

    if transform:
        return reg.predict(np.array(list(itertools.chain.from_iterable(data))).reshape(-1, 1))

    return reg.coef_[0][0], reg.intercept_[0]


def mine_dtw(series_l, series_r):
    series_l_c = series_l[:min([len(series_l), len(series_r)])]
    series_r_c = series_r[:min([len(series_l), len(series_r)])]
    return fastdtw(series_l_c, series_r_c)[0]


def get_tw_dists(prev_sample, cur_sample, window):
    dists = []

    for i in range(len(prev_sample)):
        i_0 = max([0, i - int((i / (len(prev_sample) - 1)) * window)])
        i_1 = min([window + i_0 + 1, len(prev_sample)])

        dist = mine_dtw(prev_sample[i_0: i_1], cur_sample[i_0: i_1])

        dists.append(dist)

    return dists


def time_warping(data, reg, window=36):
    to_compare = list(itertools.chain.from_iterable(data))

    return get_tw_dists(to_compare, reg, window)


def compute_correlations(feature_1, feature_2, window=36, step=1):
    corrs = []
    for i in range(window // 2, min([len(feature_1), len(feature_2)]) - window // 2, step):
        corr, _ = pearsonr(feature_1[i - window // 2: i + window // 2], feature_2[i - window // 2: i + window // 2])
        corrs.append(corr)

    return corrs
