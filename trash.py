import os
import datetime
import math
import sys
import csv
import random
from collections import Counter

import pandas as pd
from fbprophet import Prophet
import seaborn as sns
import numpy as np
import itertools

from keras.models import Sequential
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

from ssa_class import SSA


FEATURES = ['date', 'time',
             # 'head_body_distance',
   #        'body_speed_x',
   #                   'body_speed_y',
     #       'head_speed_x',
   #                        'head_speed_y',
   #                'head_body_dist_change',
   #          'head_rotation',
            # 'body_speed',
  #         'head_speed',
            #          'body_acceleration', 'head_acceleration',
            #          'body_coord_x', 'body_coord_y',
            #         'head_coord_x', 'head_coord_y',
            'temp',
            ]

CSV_PERIOD_KEY = 'new'
CSV_COMMON_PART = 'avg_600_600_'
KS_THRESH = 0.05


def load_csv(csv_name):
    with open(csv_name + '.csv', newline='') as csvfile:
        return pd.read_csv(csvfile, usecols=FEATURES)


def clear_from_nans(data):
    for key in data.keys():
        data[key] = np.nan_to_num(data[key])


def save_csv(data, csv_name):
    with open(csv_name, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(data)


def pd_to_csv(pd, csv_name):
    pd.to_csv(csv_name)


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


def average_days_by_activity(data, activity='day'):
    nights = compose_night([])
    avg_data = {
        key: [np.mean(np.array(data[key][day])[np.where(np.array(nights) == activity)].tolist()) for day in
              range(len(data[key]))]
        for key in FEATURES[2:]
    }

    return avg_data


def transform_days_data_for_pca(data, days=False, only_nights=False, activity='day'):
    if days:
        for key in FEATURES[2:]:
            data[key] = [data[key]]

    return np.array([data[key] for key in FEATURES[2:]]).T.tolist()


def pca_for_csvs_by_days(csv_for_pca, n_comp=5):
    n_comp = min([n_comp, len(FEATURES) - 2])

    pca = PCA(n_components=n_comp)

    csv_names = list(csv_for_pca.keys())

    days_count = [len(csv_for_pca[csv_name]) for csv_name in csv_names]
    indexes = [0]
    for days in days_count:
        indexes.append(indexes[-1] + days)

    indexes.append(sum(days_count))

    data = []
    for csv_name in csv_names:
        data += csv_for_pca[csv_name]

    pca.fit(data)

    pca_data = pca.transform(data).tolist()

    pca_per_csv = {csv_names[i]: pca_data[indexes[i]: indexes[i + 1]] for i in range(len(csv_names))}

    evr = pca.explained_variance_ratio_.tolist()

    return {'pca': pca_per_csv, 'evr': evr}


def plot_pca_for_csvs_by_days(pca_full_data, plot=True):
    pca_data = pca_full_data['pca']

    csv_names = list(pca_data.keys())
    pc_names = ['PC_{}'.format(i + 1) + ' {0:.2f}'.format(pca_full_data['evr'][i]) for i in
                range(len(pca_full_data['evr']))]

    concatenated_pca = []
    targets = []
    for csv_name in csv_names:
        concatenated_pca += pca_data[csv_name]
        if len(pca_data[csv_name]) > 1:
            targets += [(csv_name.split('/')[-1] + '_' + str(day)).replace(CSV_COMMON_PART, '').replace('_', ' ') for day in
                        range(len(pca_data[csv_name]))]
        else:
            #   WITHOUT DAYS
            targets += [(csv_name.split('/')[-1]).replace(CSV_COMMON_PART, '').replace('_', ' ') for day in
                        range(len(pca_data[csv_name]))]
    principalDf = pd.DataFrame(data=concatenated_pca
                               , columns=pc_names)
    targets = pd.DataFrame(data=targets, columns=['target'])

    finalDf = pd.concat([targets, principalDf], axis=1)

    if plot:
        fig = plt.figure(figsize=(14, 10))

        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel(pc_names[0], fontsize=15)
        ax.set_ylabel(pc_names[1], fontsize=15)
        ax.set_title('2D PCA', fontsize=20)

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

        targets = list(targets['target'])
        colors = cm.rainbow(np.linspace(0, 1, len(targets)))
        colors_w = list(cm.rainbow(np.linspace(0, 1, 4)))

        for target, color in zip(targets, colors):
          #  color = colors_w[int(target[-1])]

            # SH*T!!!
            if 'healthy' in target and 'april' in target:
                color = 'blue'
            if 'sick' in target and 'april' in target:
                color = 'red'
            if 'survived' in target and 'april' in target:
                color = 'green'
            if 'healthy' in target and 'june' in target:
                color = 'royalblue'
            if 'sick' in target and 'june' in target:
                color = 'orange'
            if 'survived' in target and 'june' in target:
                color = 'lime'
            if 'healthy' in target and 'july' in target:
                color = 'darkblue'
            if 'sick' in target and 'july' in target:
                color = 'firebrick'
            if 'survived' in target and 'july' in target:
                color = 'limegreen'


            # color = 'red'
            # if 'healthy' in target:
            #     color = 'green'
            # if 'survived' in target:
            #     color = 'blue'

            # if 'day' in target and 'june' in target:
            #     color = 'blue'
            # if 'day' in target and 'april' in target:
            #     color = 'red'
            # if 'night' in target and 'june' in target:
            #     color = 'black'
            # if 'night' in target and 'april' in target:
            #     color = 'green'

            # if target.split()[-1] == '1':
            #     color = 'green'
            # if target.split()[-1] == '2':
            #       color = 'blue'
            # if target.split()[-1] == '3':
            #     color = 'red'

            indicesToKeep = finalDf['target'] == target
            ax.scatter(finalDf.loc[indicesToKeep, pc_names[0]]
                       , finalDf.loc[indicesToKeep, pc_names[1]]
                       , color=color
                       , s=50)
        ax.legend(targets, prop={'size': 5}, loc=1, fancybox=True, framealpha=0.3, ncol=3, bbox_to_anchor=(1.6, 1))
        ax.grid()

        plt.show()

    return finalDf


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


def plot_feature_with_trends(data, dates=None, titles=[], title='temp', trend='ma', colors=None, save=False):
    plt.figure(figsize=(14, 10))

    plt.title(title, fontsize=15)

    if dates:
        dates_ticks = np.linspace(0, len(dates) - 1, 10)
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
    else:
        plt.show()


def create_average_healthy_day(data_by_days, days=None):
    if days:
        return {key: np.mean([data_by_days[key][day] for day in days], axis=0).tolist() for key in FEATURES[2:]}

    return {key: np.mean(data_by_days[key], axis=0).tolist() for key in FEATURES[2:]}


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
    return fastdtw(series_l, series_r)[0]


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


def opposite_period_csv(csv_name, csv_names):
    if 'healthy' in csv_name:
        name_sick = csv_name.replace('healthy', 'sick')
        name_surv = name_sick + '_survived'

        for c_n in csv_names:
            if c_n in [name_sick, name_surv]:
                return c_n
    else:
        if 'sick' in csv_name:
            name_healthy = csv_name.replace('sick', 'healthy')
        if '_survived' in csv_name:
            name_healthy = name_healthy.replace('_survived', '')

        for c_n in csv_names:
            if c_n == name_healthy:
                return c_n

    # if CSV_PERIOD_KEY in csv_name:
    #     csv_name = csv_name.replace('_', '')
    #     csv_name = csv_name.replace(CSV_PERIOD_KEY, '')
    #     for name in csv_names:
    #         sub_name = name.replace('_', '')
    #         if sub_name == csv_name and sub_name in csv_name:
    #             return name
    # else:
    #     for name in csv_names:
    #         sub_name = name.replace('_', '')
    #         if sub_name != csv_name and csv_name in sub_name:
    #             return name


def transform_data_for_hmm(data):
    concatenated = []
    windowed_data = []

    for day in data['temp']:
        concatenated += day

    window_size = 144

    for i in range(len(concatenated)):
        part = float(i) / (len(concatenated) - 1)
        to_left = int(window_size * part)
        to_right = window_size - to_left

        cur_fft = np.abs(np.fft.fft(concatenated[i - to_left: i + to_right]))[1: window_size // 2]

        windowed_data.append([np.subtract(
            *np.percentile(concatenated[i - to_left: i + to_right], [75, 25]))] +
                             np.sort(np.argsort(cur_fft[:])[-3:-1]).tolist())

    return concatenated, windowed_data


def transform_days_data_for_fft_pca(data, days=False, only_night=False):
    if days:
        for key in FEATURES[2:]:
            data[key] = [data[key]]

    current_fft_pca = []

    useful = [70, 9, 86, 71, 116, 69, 11, 74, 72, 117, 93, 6, 10, 5, 100, 94, 8, 7, 13, 15, 132, 18, 97, 31, 37, 80, 24,
              98, 109, 2, 115, 26, 120, 21, 22, 81, 3, 23, 19, 146, 79, 102, 14, 16, 110]
    for day in range(len(data['temp'])):
        current_fft_pca.append([])
        for feature in FEATURES[2:]:
            cur_fft = np.abs(np.fft.fft(data[feature][day]))[1: len(data[feature][day]) // 6]
            current_fft_pca[day] += (
                #       [np.mean(data[feature][day])] +\
                #        [np.subtract(*np.percentile(data[feature][day], [75, 25]))] +\
                cur_fft.tolist()
            )
        current_fft_pca[day] = [current_fft_pca[day][i] for i in range(len(current_fft_pca[day])) if i in useful]

    return current_fft_pca


def custom_ifft(data):
    useful = [70, 9, 86, 71, 116, 69, 11, 74, 72, 117, 93, 6, 10, 5, 100, 94, 8, 7, 13, 15, 132, 18, 97, 31, 37, 80, 24,
              98, 109, 2, 115, 26, 120, 21, 22, 81, 3, 23, 19, 146, 79, 102, 14, 16, 110]
    divided_by_feature = {}
    for f_i in range(len(FEATURES) - 2):
        divided_by_feature[FEATURES[f_i + 2]] = []
        for u in useful:
            if u // 23 == f_i:
                divided_by_feature[FEATURES[f_i + 2]].append(u % 23)
    divided_by_feature['head_body_distance'] = [divided_by_feature['head_body_distance'][0]]
    useful_feature = {}
    for feature in FEATURES[2:]:
        useful_feature[feature] = []
        for day in data[feature]:
            cur_fft = np.fft.fft(day)
            cur_fft = [cur_fft[0]] + [cur_fft[i] if (i - 1) in divided_by_feature[feature] else 0 for i in
                                      range(1, len(cur_fft))]
            cur_fft = [cur_fft[i] if (i - 1) in divided_by_feature[feature] else 0 for i in range(1, len(cur_fft))]
            useful_feature[feature].append(np.fft.ifft(cur_fft))

    return useful_feature


def detect_activity(feature_data, tresh=50):
    tresh_speed = np.percentile(feature_data, tresh)
    night_indexes = []
    for i in range(len(feature_data)):
        if i == 0:
            if np.mean(feature_data[:3]) < tresh_speed:
                night_indexes.append('night')
            else:
                night_indexes.append('day')
            continue
        if i == len(feature_data) - 1:
            if np.mean(feature_data[-3:]) < tresh_speed:
                night_indexes.append('night')
            else:
                night_indexes.append('day')
            continue

        if np.mean(feature_data[i - 1:i + 2]) < tresh_speed:
            night_indexes.append('night')
        else:
            night_indexes.append('day')

    return night_indexes


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


#    return ['night' if i in night else 'day' for i in range(24)]


def filter_feature_data(current_feature_data, date_time, speed_for_filter, night=False, tresh_max=90, tresh_min=0.02):
    tresh_speed = np.percentile(speed_for_filter, tresh_max)

    if night:
        filter_indexes = np.where(np.array(speed_for_filter) <= tresh_min * tresh_speed)
    else:
        filter_indexes = np.where(np.array(speed_for_filter) >= tresh_min * tresh_speed)

    return np.array(current_feature_data)[filter_indexes].tolist(), np.array(date_time)[filter_indexes].tolist()


def concatenate_tables(avg_by_hours_csvs, groups=[['healthy'], ['sick']], night=False):
    csv_by_keys = {}
    csv_names = list(avg_by_hours_csvs.keys())

    for key in groups:
        group_name = '_'.join(key)
        csv_by_keys[group_name] = []
        for csv_name in csv_names:
            next_csv = False
            for subkey in key:
                if subkey not in csv_name:
                    next_csv = True
                    break

            if next_csv:
                continue

            csv_by_keys[group_name].append(csv_name)

        result_dict = {}
        max_len = 0

        healthy_night = {}

        for csv_name in csv_by_keys[group_name]:
            if 'healthy' in csv_name:
                for feature in FEATURES[2:]:
                    feature_name = csv_name.split('/')[-1].replace(CSV_COMMON_PART, '').replace('_',
                                                                                                ' ') + ' ' + feature
                    result_dict[feature_name] = list(
                        itertools.chain.from_iterable(avg_by_hours_csvs[csv_name][feature]))
                    cur_len = len(result_dict[feature_name])
                    if cur_len > max_len:
                        max_len = cur_len

                if night:
                    healthy_night[csv_name] = compose_night(
                        detect_activity(list(itertools.chain.from_iterable(avg_by_hours_csvs[csv_name]['body_speed']))))

                    feature_name = csv_name.split('/')[-1].replace(CSV_COMMON_PART, '').replace('_',
                                                                                                ' ') + ' ' + 'activity'
                    result_dict[feature_name] = healthy_night[csv_name] * (
                                cur_len // 24)  # detect_activity(list(itertools.chain.from_iterable(avg_by_hours_csvs[csv_name]['body_speed'])))

        for csv_name in csv_by_keys[group_name]:
            if 'sick' in csv_name:
                for feature in FEATURES[2:]:
                    feature_name = csv_name.split('/')[-1].replace(CSV_COMMON_PART, '').replace('_',
                                                                                                ' ') + ' ' + feature
                    result_dict[feature_name] = list(
                        itertools.chain.from_iterable(avg_by_hours_csvs[csv_name][feature]))
                    cur_len = len(result_dict[feature_name])

                    if cur_len > max_len:
                        max_len = cur_len

                if night:
                    feature_name = csv_name.split('/')[-1].replace(CSV_COMMON_PART, '').replace('_',
                                                                                                ' ') + ' ' + 'activity'
                    try:
                        result_dict[feature_name] = healthy_night[opposite_period_csv(csv_name, csv_names)] * (
                                    cur_len // 24)
                    except:
                        continue

        result_dict['common dates'] = []
        for day in range(max_len // 24):
            result_dict['common dates'] += ['{} day {} hour'.format(day + 1, h + 1) for h in range(24)]

        df = pd.DataFrame.from_dict(result_dict, orient='index')
        df = df.transpose()
        pd_to_csv(df, group_name + '.csv')


def make_similar(avg_by_hour_csvs):
    csv_names = list(avg_by_hour_csvs.keys())

    max_healthy_days = 0
    for csv_name in avg_by_hour_csvs:
        if 'healthy' in csv_name:
            if len(avg_by_hour_csvs[csv_name]['temp']) > max_healthy_days:
                max_healthy_days = len(avg_by_hour_csvs[csv_name]['temp'])

    similar_data = {}
    for csv_name in avg_by_hour_csvs:
        if 'healthy' in csv_name:
            similar_data[csv_name] = {
                feature: list(itertools.chain.from_iterable(
                    (avg_by_hour_csvs[csv_name][feature] * 5)[: max_healthy_days + 1] +
                    avg_by_hour_csvs[opposite_period_csv(csv_name, csv_names)][feature]))
                for feature in FEATURES[2:]
            }

    result_dict = {}
    for csv_name in csv_names:
        if 'healthy' in csv_name:
            for feature in FEATURES[2:]:
                feature_name = csv_name.split('/')[-1].replace(CSV_COMMON_PART, '').replace('_', ' ') + ' ' + feature
                result_dict[feature_name] = similar_data[csv_name][feature]

    # for feature in FEATURES[2:]:
    #     plot_feature_with_trends([similar_data[csv_name][feature] for csv_name in csv_names if 'healthy' in csv_name],
    #                              titles=[csv_name for csv_name in csv_names if 'healthy' in csv_name],
    #                              title=feature, save=True)
    #
    # df = pd.DataFrame.from_dict(result_dict, orient='index')
    # df = df.transpose()
    # pd_to_csv(df, 'same' + '.csv')

    return similar_data


if __name__ == '__main__':
    csv_dir_names = sys.argv[1:]
    csv_dir_names = [dd.replace('test_csvs', 'norm') for dd in csv_dir_names]
    print(csv_dir_names)

    csv_names = []

    for name in csv_dir_names:
        if os.path.isdir(name):
            csv_names += sorted([os.path.join(name, d.replace('.csv', '')) for d in os.listdir(name) if '.csv' in d])
        else:
            csv_names.append(name)

    csv_names = [csv_name for csv_name in csv_names if '' in csv_name and '' in csv_name]

    for csv_name in csv_names:
        print(csv_name)

    csvs = {}
    for csv_name in csv_names:
        csvs[csv_name] = load_csv(csv_name)
        clear_from_nans(csvs[csv_name])

    by_day_csvs = {}

    for csv_name in csv_names:
        by_day_csvs[csv_name] = divide_by_days(csvs[csv_name], full_days=True)
        if 1:#'july' in csv_name:
            by_day_csvs[csv_name] = delete_bad_days(by_day_csvs[csv_name])
        by_day_csvs[csv_name] = add_mock_last_day(by_day_csvs[csv_name])

    # fft_by_table = {}
    # fft_feature = 'body_speed'
    # for csv_name in csv_names:
    #     fft_by_table[csv_name] = fft(csvs[csv_name][fft_feature], 1)[1]
    #
    # fft_df = pd.DataFrame.from_dict(fft_by_table, orient='index')
    # fft_df = fft_df.transpose()
    # pd_to_csv(fft_df, 'fft_by_table_{}.csv'.format(fft_feature))
    #
    # fft_by_day = {}
    # for csv_name in csv_names:
    #     fft_by_day[csv_name] = [
    #         fft(by_day_csvs[csv_name][fft_feature][day], len(by_day_csvs[csv_name][fft_feature][day])) for day in
    #         range(len(by_day_csvs[csv_name][fft_feature]))]
    #     fft_by_day[csv_name] = [24 / (day[0][np.argmax(day[1])] - 2) for day in fft_by_day[csv_name]]
    #     # plot_feature([{'fft by day': fft_by_day_1} for fft_by_day_1 in fft_by_day[csv_name]], 'fft by day')
    #
    # fft_df = pd.DataFrame.from_dict(fft_by_day, orient='index')
    # fft_df = fft_df.transpose()
    # pd_to_csv(fft_df, 'fft_by_day_{}.csv'.format(fft_feature))
    #
    # for feature in ['head_rotation']:
    #     for csv_name in csv_names:
    #         if 'sick' in csv_name:
    #             date_time = []
    #             dates = list(itertools.chain.from_iterable(by_day_csvs[csv_name]['date']))
    #             times = list(itertools.chain.from_iterable(by_day_csvs[csv_name]['time']))
    #             date_time = [dates[i] + ' ' + times[i] for i in range(len(dates))]
    #             dates = list(itertools.chain.from_iterable(by_day_csvs[opposite_period_csv(csv_name, csv_names)]['date']))
    #             times = list(itertools.chain.from_iterable(by_day_csvs[opposite_period_csv(csv_name, csv_names)]['time']))
    #             date_time = [dates[i] + ' ' + times[i] for i in range(len(dates))] + date_time
    #             current_feature_data = list(itertools.chain.from_iterable(by_day_csvs[opposite_period_csv(csv_name, csv_names)][feature]))
    #             current_feature_data = current_feature_data + list(itertools.chain.from_iterable(by_day_csvs[csv_name][feature]))
    #             # speed_for_filter = list(itertools.chain.from_iterable(by_day_csvs[csv_name]['body_speed']))
    #             # filtered_feature_data, filtered_dates = filter_feature_data(current_feature_data, date_time, speed_for_filter, night=False)
    #             plot_feature_with_trends([current_feature_data], date_time, title=feature + ' ' + csv_name.split('/')[-1], trend='ma', save=True)

    by_time_csvs = {}
    for csv_name in csv_names:
        by_time_csvs[csv_name] = divide_days_by_time(by_day_csvs[csv_name], 144)

    by_hours_csvs = {}
    for csv_name in csv_names:
        by_hours_csvs[csv_name] = divide_days_by_time(by_day_csvs[csv_name], 24)

    avg_by_day_csvs = {}
    for csv_name in csv_names:
        avg_by_day_csvs[csv_name] = average_by_days(by_day_csvs[csv_name])

    avg_days_csvs = {}
    for csv_name in csv_names:
        avg_days_csvs[csv_name] = average_csv_days(avg_by_day_csvs[csv_name])

    avg_by_time_csvs = {}
    for csv_name in csv_names:
        avg_by_time_csvs[csv_name] = average_days_by_time(by_time_csvs[csv_name])
        # plot_feature_with_trends([list(itertools.chain.from_iterable(avg_by_time_csvs[csv_name]['temp']))])

    avg_by_hours_csvs = {}
    for csv_name in csv_names:
        avg_by_hours_csvs[csv_name] = average_days_by_time(by_hours_csvs[csv_name])

    # make_similar(avg_by_hours_csvs)
    #
    # avg_by_day_by_day_csvs = {}
    # for csv_name in csv_names:
    #     avg_by_day_by_day_csvs[csv_name] = average_days_by_activity(avg_by_hours_csvs[csv_name])
    #     # plot_feature([{'y_coord': avg_by_day_by_day_csvs[csv_name]['body_coord_y']}], 'y_coord', titles=[csv_name + ' y coord day'])
    #
    # avg_by_night_by_day_csvs = {}
    # for csv_name in csv_names:
    #     avg_by_night_by_day_csvs[csv_name] = average_days_by_activity(avg_by_hours_csvs[csv_name], activity='night')

    # y_coord_dict = {}
    # for csv_name in csv_names:
    #     y_coord_dict[csv_name] = avg_by_night_by_day_csvs[csv_name]['body_coord_y']
    #
    # df = pd.DataFrame.from_dict(y_coord_dict, orient='index')
    # df = df.transpose()
    # pd_to_csv(df, 'body_coord_y.csv')
    #      plot_feature([{'y_coord': avg_by_night_by_day_csvs[csv_name]['body_coord_y']}], 'y_coord', titles=[csv_name + ' y coord night'])

    # avg_day_csvs = {}
    # for csv_name in csv_names:
    #     avg_day_csvs[csv_name] = average_csv_days(avg_by_day_by_day_csvs[csv_name])
    #
    # avg_night_csvs = {}
    # for csv_name in csv_names:
    #     avg_night_csvs[csv_name] = average_csv_days(avg_by_night_by_day_csvs[csv_name])

    csv_for_days_pca = {}
    for csv_name in csv_names:
        csv_for_days_pca[csv_name] = transform_days_data_for_pca(avg_by_day_csvs[csv_name].copy())

    csv_for_tables_pca = {}
    for csv_name in csv_names:
        csv_for_tables_pca[csv_name] = transform_days_data_for_pca(avg_days_csvs[csv_name].copy(), days=True)

    # csv_for_days_pca_night = {}
    # for csv_name in csv_names:
    #     csv_for_days_pca_night[csv_name] = transform_days_data_for_pca(avg_by_night_by_day_csvs[csv_name].copy())
    #
    # csv_for_tables_pca_night = {}
    # for csv_name in csv_names:
    #     csv_for_tables_pca_night[csv_name] = transform_days_data_for_pca(avg_night_csvs[csv_name].copy(), days=True)
    #
    # csv_for_days_pca_day = {}
    # for csv_name in csv_names:
    #     csv_for_days_pca_day[csv_name] = transform_days_data_for_pca(avg_by_day_by_day_csvs[csv_name].copy())
    #
    # csv_for_tables_pca_day = {}
    # for csv_name in csv_names:
    #     csv_for_tables_pca_day[csv_name] = transform_days_data_for_pca(avg_day_csvs[csv_name].copy(), days=True)

    # csv_for_days_fft_pca = {}
    # for csv_name in csv_names:
    #     csv_for_days_fft_pca[csv_name] = transform_days_data_for_fft_pca(avg_by_time_csvs[csv_name].copy())
    #
    # fft_dict = {}
    # float_fft_table = []
    # for csv_name in csv_names:
    #     for day in range(len(csv_for_days_fft_pca[csv_name])):
    #         fft_name = csv_name.split('/')[-1] + '_' + str(day)
    #         fft_dict[fft_name] = csv_for_days_fft_pca[csv_name][day]
    #
    #         float_fft_table.append(fft_dict[fft_name])
    #
    # fft_dict['model'] = [70, 9, 86, 71, 116, 69, 11, 74, 72, 117, 93, 6, 10, 5, 100, 94, 8, 7, 13, 15, 132, 18, 97, 31, 37, 80, 24,
    #           98, 109, 2, 115, 26, 120, 21, 22, 81, 3, 23, 19, 146, 79, 102, 14, 16, 110]
    #
    # float_fft_table = np.array(float_fft_table).T.tolist()
    # float_fft_table = {str(fft_dict['model'][i]): [float_fft_table[i]] for i in range(len(fft_dict['model']))}

    # pca_data = pca_for_csvs_by_days(float_fft_table)
    # pd_to_csv(plot_pca_for_csvs_by_days(pca_data, plot=True), 'pca_for_fft_freqs.csv')
    #
    # df = pd.DataFrame.from_dict(fft_dict, orient='index')
    # df = df.transpose()
    # df = df.set_index('model')
    # del df.index.name

    # pd_to_csv(df, 'useful_freqs_fft_table' + '.csv')
    #
    # sns.clustermap(df, metric="euclidean", standard_scale=1, method="ward")
    # sns.clustermap(df, metric="correlation", standard_scale=1, method="single")
    # plt.show()

    # custom_ifft_by_days = {}
    # for csv_name in csv_names:
    #     custom_ifft_by_days[csv_name] = custom_ifft(avg_by_time_csvs[csv_name].copy())
    #
    # freq_fft_table = {}
    # counter = 0
    # max_len = 0
    # for csv_name in csv_names:
    #     if 'healthy' in csv_name:
    #         for feature in ['temp']:  # FEATURES[2:]:
    #             h_ifft = list(itertools.chain.from_iterable(custom_ifft_by_days[csv_name][feature]))
    #             if len(h_ifft) > max_len:
    #                 max_len = len(h_ifft)
    #
    # for csv_name in csv_names:
    #     if 'healthy' in csv_name:
    #         counter += 0.1
    #         for feature in ['temp']:  # FEATURES[2:]:
    #             h_ifft = (list(itertools.chain.from_iterable(custom_ifft_by_days[csv_name][feature])) * 10)[: max_len]
    #             s_ifft = list(itertools.chain.from_iterable(
    #                 custom_ifft_by_days[opposite_period_csv(csv_name, csv_names)][feature]))
    #             ifft_together = (np.array(h_ifft + s_ifft) + counter).tolist()
    #             h_original = list(itertools.chain.from_iterable(avg_by_time_csvs[csv_name][feature]))
    #             s_original = list(
    #                 itertools.chain.from_iterable(avg_by_time_csvs[opposite_period_csv(csv_name, csv_names)][feature]))
    #             original_together = (np.array(h_original + s_original) + counter).tolist()
    #
    #         #     plot_feature([{
    #         #         feature: np.real(h_ifft).tolist(),
    #         #     }, {
    #         #         feature: h_original + s_original,
    #         #     }], feature, titles=[csv_name.split('/')[-1] + ' ' + feature + ' ifft', 'original'])
    #         # freq_fft_table[csv_name.split('/')[-1].replace('_', ' ').replace('healthy', '') + ' fft'] = np.real(ifft_together).tolist()
    #
    # df = pd.DataFrame.from_dict(freq_fft_table, orient='index')
    # df = df.transpose()
    # pd_to_csv(df, 'freq_fft_{}.csv'.format(feature))
    #
    # csv_for_tables_fft_pca = {}
    # for csv_name in csv_names:
    #     csv_for_tables_fft_pca[csv_name] = transform_days_data_for_fft_pca(avg_by_time_csvs[csv_name].copy(), days=True)
    #
    # concatenate_tables(avg_by_hours_csvs.copy(), groups=[['april'], ['june']], night=True)
    #
    # pca_data = pca_for_csvs_by_days(csv_for_days_pca)
    # pd_to_csv(plot_pca_for_csvs_by_days(pca_data, plot=True), 'pca_for_days.csv')

    # pca_data = pca_for_csvs_by_days(csv_for_days_fft_pca)
    # pd_to_csv(plot_pca_for_csvs_by_days(pca_data, plot=True), 'pca_for_fft_days.csv')
    #
    # pca_data = pca_for_csvs_by_days(csv_for_tables_fft_pca)
    # pd_to_csv(plot_pca_for_csvs_by_days(pca_data, plot=True), 'pca_for_fft_individuals.csv')

    # pca_data = pca_for_csvs_by_days(csv_for_tables_pca)
    # pd_to_csv(plot_pca_for_csvs_by_days(pca_data, plot=True), 'pca_for_individuals.csv')

    # # NIGHTS!!!!!!!!!!
    # pca_data = pca_for_csvs_by_days(csv_for_days_pca_day)
    # pd_to_csv(plot_pca_for_csvs_by_days(pca_data, plot=True), 'pca_for_days_day.csv')

    # pca_data = pca_for_csvs_by_days(csv_for_tables_pca_day)
    # pd_to_csv(plot_pca_for_csvs_by_days(pca_data, plot=True), 'pca_for_individuals_day.csv')
    #
    # # DAYS AND NIGHTS!!!!!!!!!!
    # day_night_pca_days_dict = {}
    # for key in csv_for_days_pca_day.keys():
    #     print(key+'_day')
    #     day_night_pca_days_dict[key + '_day'] = csv_for_days_pca_day[key]
    # for key in csv_for_days_pca_night.keys():
    #     day_night_pca_days_dict[key + '_night'] = csv_for_days_pca_night[key]
    #
    # pca_data = pca_for_csvs_by_days(day_night_pca_days_dict)
    # pd_to_csv(plot_pca_for_csvs_by_days(pca_data, plot=True), 'pca_for_days_days_night.csv')

    # day_night_pca_tables_dict = {}
    # for key in csv_for_tables_pca_day.keys():
    #     day_night_pca_tables_dict[key + '_day'] = csv_for_tables_pca_day[key]
    # for key in csv_for_tables_pca_night.keys():
    #     day_night_pca_tables_dict[key + '_night'] = csv_for_tables_pca_night[key]
    #
    # pca_data = pca_for_csvs_by_days(day_night_pca_tables_dict)
    # pd_to_csv(plot_pca_for_csvs_by_days(pca_data, plot=True), 'pca_for_individuals_days_night.csv')

    # all_healthy_days = {feature: [] for feature in FEATURES[2:]}
    # for csv_name in csv_names:
    #     if 'healthy' in csv_name:
    #         for feature in FEATURES[2:]:
    #             all_healthy_days[feature] += avg_by_time_csvs[csv_name][feature][:-1]
    #
    # avg_healthy_day = create_average_healthy_day(all_healthy_days)
    # for feature in FEATURES[2:]:
    #     avg_hea
    avg_healthy_day = {}
    for csv_name in csv_names:
        if 'healthy' in csv_name:
            avg_healthy_day[csv_name] = create_average_healthy_day({feature: avg_by_time_csvs[csv_name][feature] for feature in FEATURES[2:]})
            avg_healthy_day[csv_name] = {feature: avg_healthy_day[csv_name][feature] * 20 for feature in FEATURES[2:]}
    # regressions = {}
    # for csv_name in csv_names:
    #     if 'sick' in csv_name or 'survived' in csv_name:
    #         try:
    #             print(csv_name, opposite_period_csv(csv_name, csv_names))
    #             regressions[csv_name] = regression(avg_by_time_csvs[csv_name]['temp'], avg_by_time_csvs[opposite_period_csv(csv_name, csv_names)]['temp'])
    #             plot_feature([{'reg': regressions[csv_name]},
    #                           {'reg': list(itertools.chain.from_iterable(avg_by_time_csvs[csv_name]['temp']))},
    #                           {'reg': list(itertools.chain.from_iterable(avg_by_time_csvs[opposite_period_csv(csv_name, csv_names)]['temp']))},
    #                           ], 'reg')
    #         except:
    #             import traceback
    #             traceback.print_exc()
    #

    date_times = {}
    for csv_name in csv_names:
        if 'sick' in csv_name:
            try:
                dates = avg_by_time_csvs[opposite_period_csv(csv_name, csv_names)]['date']
                times = avg_by_time_csvs[opposite_period_csv(csv_name, csv_names)]['time']
            except:
                dates = []
                times = []
                continue
            dates += avg_by_time_csvs[csv_name]['date']
            times += avg_by_time_csvs[csv_name]['time']
            dates = list(itertools.chain.from_iterable(dates))
            times = list(itertools.chain.from_iterable(times))
            csv_name = csv_name.split('/')[-1].replace('sick', '')
            date_times[csv_name] = [dates[i] + ' ' + times[i] for i in range(len(dates))]

    time_warpings_all_features = {}
    for feature in FEATURES[2:]:
        decompose_table = {}
        decompose_original = {}
        decompose_trend = {}
        decompose_season = {}
        decompose_resid = {}
        time_warpings = {}
        max_tw = 0
        min_tw = 0
        min_tw_length = 1005000
        tw_table = {}
        counter = 2
        decompose_pca_trend = {}
        decompose_pca_season = {}
        decompose_pca_resid = {}
        fft_trend_pca = {}
        healthy_length = {}
        for csv_name in csv_names:
            if 'sick' in csv_name:
                try:
                    to_compare = (avg_by_time_csvs[opposite_period_csv(csv_name, csv_names)][feature] * 6)[: 6]
                except:
                    to_compare = []
                    continue
                healthy_length[csv_name] = len(to_compare) - 1
                to_compare += avg_by_time_csvs[csv_name][feature]
                ssa = SSA(list(itertools.chain.from_iterable(to_compare)), 72)

                plt.figure(figsize=(14, 8))
                plt.title(csv_name + ' ' + feature)

                # plt.plot(moving_average(list(itertools.chain.from_iterable(to_compare)), 10), label='original', linewidth=1)
                # plt.plot(moving_average(list(itertools.chain.from_iterable(to_compare))), label='moving avg.', linewidth=3)
                #
                # plt.plot(ssa.TS_comps[:, 0], label='SSA 0', linewidth=3)
                plt.plot(ssa.TS_comps[:, 1], label='SSA 1')
                plt.plot(ssa.TS_comps[:, 2], label='SSA 2')
                plt.plot(ssa.TS_comps[:, 3], label='SSA 3')

                plt.legend()
                plt.show()

                # result = seasonal_decompose(list(itertools.chain.from_iterable(to_compare)), model='additive', freq=144)

                # if 'july' in csv_name:
                #     result.plot()
               # plt.savefig('plots/SD/12h_{}_{}.png'.format(csv_name.split('/')[-1].replace('sick', ''), feature))
                # result = seasonal_decompose(list(itertools.chain.from_iterable(to_compare)), model='multiplicative', freq=144)
                # result.plot()
                #     plt.show()
                # decompose_table[csv_name + ' original'] = list(itertools.chain.from_iterable(to_compare))
                # decompose_table[csv_name + ' trend'] = result.trend.tolist()
                # if '' in csv_name:
                #     csv_name = csv_name.split('/')[-1].replace('sick', '')
                #     decompose_trend[csv_name] = result.trend.tolist()[144: -144]
                #
                #     # decompose_original[csv_name] = np.array(np.array_split(decompose_trend[csv_name], len(decompose_trend[csv_name]) // 144)).tolist()
                #     decompose_original[csv_name] = list(itertools.chain.from_iterable(to_compare))

                    # df = pd.DataFrame.from_dict({'ds': [(datetime.date.today() + datetime.timedelta(days=d)).strftime("%Y-%m-%d") for d in range(len(decompose_original[csv_name]))][:-144], 'y': decompose_original[csv_name][:-144]}, orient='index')
                    # df = pd.DataFrame.from_dict({'ds': , 'y': decompose_original[csv_name][:-144]}, orient='index')
                    # df = df.transpose()
                    #
                    # datetimes = [(datetime.date.today() + datetime.timedelta(days=d)) for d in range(len(decompose_original[csv_name]))]
                    # m = Prophet()
                    # m.fit(df)
                    # future = m.make_future_dataframe(periods=144)
                    # forecast = m.predict(future)
                    # figsize = (10, 6)
                    # xlabel = 'ds'
                    # ylabel = 'y'
                    # fig = plt.figure(facecolor='w', figsize=figsize)
                    # ax = fig.add_subplot(111)
                    # ax.plot(m.history['ds'].dt.to_pydatetime(), m.history['y'], 'k.')
                    # ax.plot(datetimes, moving_average(decompose_original[csv_name]), color='red')
                    # fcst_t = forecast['ds'].dt.to_pydatetime()
                    # ax.plot(fcst_t, forecast['yhat'], ls='-', c='#0072B2')
                    # ax.fill_between(fcst_t, forecast['yhat_lower'], forecast['yhat_upper'], color='#0072B2', alpha=0.2)
                    # ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
                    # ax.set_xlabel(xlabel)
                    # ax.set_ylabel(ylabel)
                    # fig.tight_layout()
                    # plt.show()

                    # print(len(decompose_original[csv_name]), len(date_times[csv_name]))
                    # decompose_original[csv_name + ' datetime'] = date_times[csv_name]
                    # decompose_trend[csv_name] = np.array(np.array_split(decompose_trend[csv_name], len(decompose_trend[csv_name]) // 144)).tolist()


                #     # trend_fft_h = np.abs(np.fft.fft(list(itertools.chain.from_iterable(decompose_trend[csv_name][:4]))))
                #     # trend_fft_h = trend_fft_h[1: 37]
                #     # trend_fft_h = (trend_fft_h - np.min(trend_fft_h)) / (np.max(trend_fft_h) - np.min(trend_fft_h))
                #     # trend_fft_s = np.abs(np.fft.fft(list(itertools.chain.from_iterable(decompose_trend[csv_name][-4:]))))
                #     # trend_fft_s = trend_fft_s[1: 37]
                #     # trend_fft_s = (trend_fft_s - np.min(trend_fft_s)) / (np.max(trend_fft_s) - np.min(trend_fft_s))
                #     # fft_trend_pca[csv_name + '_1'] = [trend_fft_h]
                #     # fft_trend_pca[csv_name + '_3'] = [trend_fft_s]
                #     #
                #     # # plt.figure(figsize=(14, 8))
                #     # plt.title(csv_name + ' ' + feature)
                #     # plt.plot(trend_fft_h, color='red')
                #     # plt.plot(trend_fft_s, color='blue')
                #     # plt.show()
                #
                #     # decompose_trend[csv_name + '_1'] = decompose_trend[csv_name][:4]
                #     # decompose_trend[csv_name + '_2'] = decompose_trend[csv_name][4:-4]
                #     # decompose_trend[csv_name + '_3'] = decompose_trend[csv_name][-4:]
                #     # del decompose_trend[csv_name]
                #     #
                #     decompose_season[csv_name] = [result.seasonal.tolist()[:144]]
                #
                #     decompose_resid[csv_name] = result.resid.tolist()[144: -144]
                #     decompose_resid[csv_name] = np.array(
                #         np.array_split(decompose_resid[csv_name], len(decompose_resid[csv_name]) // 144)).tolist()
                #     decompose_resid[csv_name + '_1'] = decompose_resid[csv_name][:4]
                #     decompose_resid[csv_name + '_2'] = decompose_resid[csv_name][4:-4]
                #     decompose_resid[csv_name + '_3'] = decompose_resid[csv_name][-4:]
                #     del decompose_resid[csv_name]

                # decompose_table[csv_name + ' trend'][:80] = reversed(decompose_table[csv_name + ' trend'][80: 160])
                # decompose_table[csv_name + ' trend'][-80:] = reversed(decompose_table[csv_name + ' trend'][-160: -80])
                # decompose_table[csv_name + ' seasonal'] = result.seasonal.tolist()[:144]
                # decompose_table[csv_name + ' resid'] = result.resid.tolist()
                # decompose_table[csv_name + ' resid'][:80] = reversed(decompose_table[csv_name + ' resid'][80: 160])
                # decompose_table[csv_name + ' resid'][-80:] = reversed(decompose_table[csv_name + ' resid'][-160: -80])

                # decompose_pca_trend[csv_name] = np.array_split(decompose_table[csv_name + ' trend'], len(decompose_table[csv_name + ' trend']) // 144)[1:-1]
                # decompose_pca_season[csv_name] = [decompose_table[csv_name + ' seasonal']]
                # decompose_pca_resid[csv_name] = np.array_split(decompose_table[csv_name + ' resid'], len(decompose_table[csv_name + ' resid']) // 144)[1: -1]

        # max_days = 0
        # for csv_name in decompose_trend.keys():
        #     if len(decompose_trend[csv_name]) > max_days:
        #         max_days = len(decompose_trend[csv_name])
        #
        # for day in range(max_days):
        #     plot_feature_with_trends(
        #         [decompose_trend[csv_name][day] if day < len(decompose_trend[csv_name]) else [] for csv_name in decompose_trend.keys()],
        #         title='april 24hr {} trend day {}'.format(feature, day),
        #         trend='SD',
        #         colors=['green' if 'survived' in csv_name else 'red' for csv_name in decompose_trend.keys()],
        #         save=True
        #     )

        # pca_data = pca_for_csvs_by_days(decompose_trend)
        # pd_to_csv(plot_pca_for_csvs_by_days(pca_data, plot=True), 'SD_pca_{}_trend.csv'.format(feature))
        # pca_data = pca_for_csvs_by_days(decompose_season)
        # pd_to_csv(plot_pca_for_csvs_by_days(pca_data, plot=True), 'SD_pca_{}_season.csv'.format(feature))
        # pca_data = pca_for_csvs_by_days(decompose_resid)
        # pd_to_csv(plot_pca_for_csvs_by_days(pca_data, plot=True), 'SD_pca_{}_resid.csv'.format(feature))
        # pca_data = pca_for_csvs_by_days(decompose_pca_season)
        # pd_to_csv(plot_pca_for_csvs_by_days(pca_data, plot=True), 'SD_pca_{}_season.csv'.format(feature))
        # pca_data = pca_for_csvs_by_days(fft_trend_pca)
        # pd_to_csv(plot_pca_for_csvs_by_days(pca_data, plot=True), 'april_SD_fft_pca_{}_trend.csv'.format(feature))
                with_compare = avg_healthy_day[opposite_period_csv(csv_name, csv_names)][feature][: len(to_compare) * len(to_compare[0]) + 1]

                       # with_compare = (np.array(with_compare) + counter).tolist()

                time_warpings[csv_name] = time_warping(to_compare, with_compare)
        #
        #         if np.max(time_warpings[csv_name]) - np.min(time_warpings[csv_name]) > max_tw - min_tw:
        #             min_tw = np.min(time_warpings[csv_name])
        #             max_tw = np.max(time_warpings[csv_name])
        #         if len(time_warpings[csv_name]) < min_tw_length:
        #             min_tw_length = len(time_warpings[csv_name])

        # new_tw_dict = {}
        # for csv_name in time_warpings.keys():
        #     days = np.array_split(time_warpings[csv_name], len(time_warpings[csv_name]) // 144)
        #     for i, day in enumerate(days):
        #         new_tw_dict[csv_name.split('/')[-1].replace('sick', '') + '_day_{}'.format(i + 1)] = day.tolist()
        #
        # cols = list(new_tw_dict.keys())
        # cols.sort(key=lambda x: int(x.split('_')[-1]))
        # df = pd.DataFrame.from_dict(new_tw_dict, orient='index')
        # df = df.transpose()
        # df.to_csv('TW_{}_by_day.csv'.format(feature), columns=cols)

        # cols = list(decompose_original.keys())
        # df = pd.DataFrame.from_dict(decompose_original, orient='index')
        # df = df.transpose()
        # df.to_csv('combined_{}.csv'.format(feature), columns=cols)
        #
        # new_sd_dict = {}
        # for csv_name in decompose_trend.keys():
        #     days = decompose_trend[csv_name]
        #     for i, day in enumerate(days):
        #         new_sd_dict[csv_name.split('/')[-1].replace('sick', '') + '_day_{}'.format(i + 1)] = day
        #
        # cols = list(new_sd_dict.keys())
        # cols.sort(key=lambda x: int(x.split('_')[-1]))
        # df = pd.DataFrame.from_dict(new_sd_dict, orient='index')
        # df = df.transpose()
        # df.to_csv('SD_trend_{}_by_day.csv'.format(feature), columns=cols)
        #
        # new_sd_dict = {}
        # for csv_name in decompose_season.keys():
        #     days = decompose_season[csv_name]
        #     for i, day in enumerate(days):
        #         new_sd_dict[csv_name.split('/')[-1].replace('sick', '')] = day
        #
        # cols = list(new_sd_dict.keys())
        # df = pd.DataFrame.from_dict(new_sd_dict, orient='index')
        # df = df.transpose()
        # df.to_csv('SD_season_{}.csv'.format(feature), columns=cols)
        #
        # new_sd_dict = {}
        # for csv_name in decompose_resid.keys():
        #     days = decompose_resid[csv_name]
        #     for i, day in enumerate(days):
        #         new_sd_dict[csv_name.split('/')[-1].replace('sick', '') + '_day_{}'.format(i + 1)] = day
        #
        # cols = list(new_sd_dict.keys())
        # cols.sort(key=lambda x: int(x.split('_')[-1]))
        # df = pd.DataFrame.from_dict(new_sd_dict, orient='index')
        # df = df.transpose()
        # df.to_csv('SD_resid_{}_by_day.csv'.format(feature), columns=cols)

        # df = pd.DataFrame.from_dict(decompose_table, orient='index')
        # df = df.transpose()
        # pd_to_csv(df, '12h_SD_{}.csv'.format(feature))

        # for csv_name in time_warpings.keys():
        #     if csv_name not in time_warpings_all_features.keys():
        #         time_warpings_all_features[csv_name] = []
        #     time_warpings_all_features[csv_name] += time_warpings[csv_name][-min_tw_length:]

        # clust_model = KMeans(n_clusters=4)
        # predictions = clust_model.fit_predict(
        #     [time_warpings[csv_name][-min_tw_length:] for csv_name in time_warpings.keys()])
        # predictions = {[csv_name for csv_name in time_warpings.keys()][i]: predictions[i] for i in
        #                range(len(predictions))}
        #
        # tw_pca_data = {csv_name + ' ' + str(predictions[csv_name]): [time_warpings[csv_name][-min_tw_length:]] for
        #                csv_name in time_warpings.keys()}
        # pca_data = pca_for_csvs_by_days(tw_pca_data)
        # pd_to_csv(plot_pca_for_csvs_by_days(pca_data, plot=False), 'tw_pca_{}.csv'.format(feature))
        #
        # tw_pca_data = {csv_name: np.array_split(time_warpings[csv_name], len(time_warpings[csv_name]) // 144)
        #                for csv_name in time_warpings.keys()}
        # pca_data = pca_for_csvs_by_days(tw_pca_data)
        # pd_to_csv(plot_pca_for_csvs_by_days(pca_data, plot=True), 'tw_pca_{}_by_day.csv'.format(feature))

        trends_tw = {}
        for csv_name in csv_names:
            if 'sick' in csv_name:
                counter += 2
                try:
                    to_compare = (avg_by_time_csvs[opposite_period_csv(csv_name, csv_names)][feature]* 6)[: 6]
                except:
                    to_compare = []
                    continue
                # to_compare = avg_by_time_csvs[csv_name][feature]
                # # to_compare = (np.array(to_compare) + counter).tolist()
                #
                # with_compare = avg_healthy_day[opposite_period_csv(csv_name, csv_names)][feature][: len(to_compare) * len(to_compare[0]) + 1]

                # with_compare = (np.array(with_compare) + counter).tolist()

                result = seasonal_decompose(time_warpings[csv_name], model='additive', freq=144)
                # plt.figure(figsize=(14, 10))
                # plt.title(csv_name.split('/')[-1].replace('sick', '') + '_' + feature)
                # plt.plot(time_warpings[csv_name][72: -72])
                # plt.plot(result.trend[72:-72])
                # # plt.show()
                # plt.savefig('plots/trends/tw/{}.png'.format(csv_name.split('/')[-1].replace('sick', '') + '_' + feature))
                trend_tw = result.trend.tolist()[144: -144]
                # trends_tw[csv_name] = np.array_split(trend_tw, len(trend_tw) // 144)
                trends_tw[csv_name] = trend_tw
        #
        # labels = []
        # tw_trend_clust_data = []
        # print(healthy_length)
        # for csv_name in trends_tw.keys():
        #     labels += (['H_{}'.format(day) for day in range(healthy_length[csv_name])] + ['S_{}'.format(day) for day in range(len(trends_tw[csv_name]) - healthy_length[csv_name])])
        #     tw_trend_clust_data += trends_tw[csv_name]
        #
        # cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
        # cluster.fit_predict(tw_trend_clust_data)
        #
        # pca = PCA(n_components=3)
        # pca.fit(tw_trend_clust_data)
        # pca_data = pca.transform(tw_trend_clust_data)
        #
        # # pca_tw_trend = pca_for_csvs_by_days(trends_tw)['pca']
        # fig = plt.figure(figsize=(10, 7))
        # ax = fig.add_subplot(1, 1, 1)
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
        # predictions = cluster.labels_.tolist()
        # colors_w = list(cm.rainbow(np.linspace(0, 1, 3)))
        # for i in range(len(pca_data)):
        #     ax.scatter([pca_data[i, 0]], [pca_data[i, 1]], color=colors_w[predictions[i]], s=50)
        # print(labels)
        # ax.legend(labels, prop={'size': 5}, loc=1, fancybox=True, framealpha=0.3, ncol=3, bbox_to_anchor=(1., 1))
        # ax.grid()
        # plt.show()
        #
        # evr = pca.explained_variance_ratio_.tolist()
        # pc_names = ['PC_{}'.format(i + 1) + ' {0:.2f}'.format(evr[i]) for i in
        #             range(len(evr))]
        # principalDf = pd.DataFrame(data=pca_data
        #                            , columns=pc_names)
        # targets = pd.DataFrame(data=labels, columns=['target'])
        # finalDf = pd.concat([targets, principalDf], axis=1)
        # pd_to_csv(finalDf, 'tw_trend_{}.csv'.format(feature))

        # linked = linkage(tw_trend_clust_data, 'single')
        #
        # labelList = labels
        #
        # plt.figure(figsize=(10, 7))
        # dendrogram(linked,
        #            orientation='top',
        #            labels=labelList,
        #            distance_sort='descending',
        #            show_leaf_counts=True)
        # plt.show()
        #         time_warpings[csv_name] = (((np.array(time_warpings[csv_name]) - np.min(time_warpings[csv_name])) / (max_tw - min_tw)) + counter).tolist()
        #
        #
        #         # plot_feature([{'reg': time_warpings[csv_name]},
        #         #               {'reg': list(itertools.chain.from_iterable(to_compare))},
        #         #               {'reg': with_compare},
        #         #               ], 'reg',
        #         #              titles=['time warping for ' + feature + ' ' + csv_name.split('/')[-1].replace('sick', ''),
        #         #                      'original (H + S)', 'Avg. H'])
        #
        #         tw_table[csv_name.split('/')[-1].replace('_', ' ').replace('sick', '') + ' tw'] = time_warpings[csv_name]
        #         tw_table[csv_name.split('/')[-1].replace('_', ' ').replace('sick', '') + ' original (H. + S.)'] = list(itertools.chain.from_iterable(to_compare))
        #         tw_table[csv_name.split('/')[-1].replace('_', ' ').replace('sick', '') + ' avg H.'] = with_compare

        # df = pd.DataFrame.from_dict(trends_tw, orient='index')
        # df = df.transpose()
        # pd_to_csv(df, 'trend_tw_{}.csv'.format(feature))

    # tw_pca_data = {csv_name: [time_warpings_all_features[csv_name]] for
    #                csv_name in time_warpings_all_features.keys()}
    # pca_data = pca_for_csvs_by_days(tw_pca_data)
    # pd_to_csv(plot_pca_for_csvs_by_days(pca_data, plot=True), 'tw_pca_all_features_only_sick.csv')

    # tw_pca_data = {csv_name + ' ' + str(predictions[csv_name]): np.array_split(time_warpings_all_features[csv_name],
    #                                                                            len(time_warpings_all_features[csv_name]) // 144)
    #                for csv_name in time_warpings.keys()}
    # pca_data = pca_for_csvs_by_days(tw_pca_data)
    # pd_to_csv(plot_pca_for_csvs_by_days(pca_data, plot=False), 'tw_pca_{}_by_day.csv'.format(feature))

    #
    # feature = 'head_rotation'
    # for csv_name in csv_names:
    #     ks_scores = kstest(avg_by_time_csvs[csv_name],
    #                        csv_average_healthy_days[opposite_period_csv(csv_name, csv_names)],
    #                        feature,
    #                        csv_name, opposite_period_csv(csv_name, csv_names))
    #
    #     pd_to_csv(ks_scores, '{}_hr_kstest.csv'.format(csv_name.split('/')[-1].replace(CSV_COMMON_PART, '')))
    #     plot_feature([ks_scores], 'ks_scores', titles=['ks test, {} for {}'.format(feature, csv_name.split('/')[-1].replace(CSV_COMMON_PART, ''))], save=True)
    #
    # targets = []
    # for csv_name in csv_names:
    #     if 'healthy' in csv_name:
    #         targets += [1] * len(csv_for_days_pca_day[csv_name])
    #     elif 'survived' in csv_name:  #(CSV_PERIOD_KEY in csv_name and 'lonely' in csv_name) or ('sick_2_ucsf' in csv_name or 'sick_1_ucsf' in csv_name) or ('sick_2_pc' in csv_name or 'sick_3_pc' in csv_name):
    #         targets += [2] * len(csv_for_days_pca_day[csv_name])
    #     elif 'sick' in csv_name:
    #         targets += [3] * len(csv_for_days_pca_day[csv_name])
    #
    # data = []
    # for csv_name in csv_names:
    #     data += csv_for_days_pca_day[csv_name]
    # data = pd.DataFrame(data=data, columns=FEATURES[2:])
    # targets = pd.DataFrame(data=targets, columns=['target'])
    # data = pd.concat([data, targets], axis=1)
    #
    # pd_to_csv(random_forest(data), 'rf_features_importances_day.csv')
    #
    # targets = []
    # for csv_name in csv_names:
    #     if 'healthy' in csv_name:
    #         targets += [1] * len(csv_for_days_fft_pca[csv_name])
    #     elif 'survived' in csv_name:  # (CSV_PERIOD_KEY in csv_name and 'lonely' in csv_name) or ('sick_2_ucsf' in csv_name or 'sick_1_ucsf' in csv_name) or ('sick_2_pc' in csv_name or 'sick_3_pc' in csv_name):
    #         targets += [2] * len(csv_for_days_fft_pca[csv_name])
    #     elif 'sick' in csv_name:
    #         targets += [3] * len(csv_for_days_fft_pca[csv_name])
    #
    # data = []
    # for csv_name in csv_names:
    #     data += csv_for_days_fft_pca[csv_name]
    # data = pd.DataFrame(data=data, columns=FEATURES[2:])
    # targets = pd.DataFrame(data=targets, columns=['target'])
    # data = pd.concat([data, targets], axis=1)
    #
    # pd_to_csv(random_forest(data), 'rf_features_importances_intq.csv')

    # n_comp = 6
    # colors = cm.rainbow(np.linspace(0, 1, n_comp))
    #
    # csv_data_hmm = {}
    # for csv_name in decompose_trend.keys():
    #     csv_data_hmm[csv_name] = transform_data_for_hmm({'temp': [decompose_trend[csv_name]]})
    #
    # model = hmm.GaussianHMM(n_components=n_comp)
    #
    # train_data = []
    # for csv_name in decompose_trend.keys():
    #     train_data += csv_data_hmm[csv_name][1]
    #
    # model.fit(train_data)
    #
    # for csv_name in decompose_trend.keys():
    #     print(csv_name)
    #     states = model.predict(csv_data_hmm[csv_name][1])
    #
    #     plt.figure(figsize=(14, 8))
    #     plt.title('HMM states for ' + csv_name)
    #     for i in range(len(states)):
    #         plt.plot(i, csv_data_hmm[csv_name][0][i], color=colors[states[i]], marker='.', markersize=10)
    #
    #     plt.ylim([0.75 * np.min(csv_data_hmm[csv_name][0]), 1.25 * np.max(csv_data_hmm[csv_name][0])])
    #
    #     plt.savefig('plots/hmm/hmm_{}.png'.format(csv_name.replace('/', '_')))

    # decompose_trend_by_csv = {}
    # for csv_name in csv_names:
    #     if len(avg_by_time_csvs[csv_name]['temp']) <= 2:
    #         continue
    #     decompose_trend_by_csv[csv_name] = {}
    #     for feature in FEATURES[2:]:
    #         result = seasonal_decompose(
    #             list(itertools.chain.from_iterable(avg_by_time_csvs[csv_name][feature])),
    #             model='additive',
    #             freq=144
    #         )
    #         trend = result.trend.tolist()[72: -72]
    #         decompose_trend_by_csv[csv_name][feature] = trend
    #         # decompose_trend_by_csv[csv_name][feature] = np.array(np.array_split(trend, len(trend) // 144)).tolist()
    #
    # labels = {'healthy': [1, 0, 0], 'survived': [0, 1, 0], 'sick': [0, 0, 1]}
    # dataset_by_class = {
    #     'healthy': {},
    #     'sick': {},
    #     'survived': {}
    # }
    # test_dataset_by_class = {
    #     'healthy': {},
    #     'sick': {},
    #     'survived': {}
    # }
    # dataset_y = {'healthy': [], 'sick': [], 'survived': []}
    # test_dataset_y = {'healthy': [], 'sick': [], 'survived': []}
    #
    # counter = []
    # for csv_name in decompose_trend_by_csv.keys():
    #     test_dataset_by_class['healthy'][csv_name] = [[] for feature in FEATURES[2:]]
    #     test_dataset_by_class['sick'][csv_name] = [[] for feature in FEATURES[2:]]
    #     test_dataset_by_class['survived'][csv_name] = [[] for feature in FEATURES[2:]]
    #     dataset_by_class['healthy'][csv_name] = [[] for feature in FEATURES[2:]]
    #     dataset_by_class['sick'][csv_name] = [[] for feature in FEATURES[2:]]
    #     dataset_by_class['survived'][csv_name] = [[] for feature in FEATURES[2:]]
    #
    #     print(csv_name.split('/')[-1])
    #     if 'survived' in csv_name:#'april_healthy_1_pc' in csv_name or 'june_healthy_2_ucsf' in csv_name or 'april_healthy_3_ucsf' in csv_name or 'july_sick_3_ucsf' in csv_name or 'july_sick_2_ucsf' in csv_name or 'june_sick_2_pc' in csv_name or 'june_sick_3_ucsf' in csv_name or 'june_sick_3_ucsf' in csv_name:
    #         for feature in FEATURES[2:]:
    #             chunk = decompose_trend_by_csv[csv_name][feature]
    #             if 'healthy' in csv_name:
    #                 test_dataset_by_class['healthy'][csv_name][FEATURES.index(feature) - 2] += chunk
    #                 continue
    #             if 'survived' in csv_name:
    #                 test_dataset_by_class['survived'][csv_name][FEATURES.index(feature) - 2] += chunk
    #                 continue
    #             if 'sick' in csv_name:
    #                 test_dataset_by_class['sick'][csv_name][FEATURES.index(feature) - 2] += chunk
    #                 continue
    #     else:
    #         for feature in FEATURES[2:]:
    #             chunk = decompose_trend_by_csv[csv_name][feature]
    #             if 'healthy' in csv_name:
    #                 dataset_by_class['healthy'][csv_name][FEATURES.index(feature) - 2] += chunk
    #                 counter += ['healthy'] * len(chunk)
    #                 continue
    #             if 'survived' in csv_name:
    #                 dataset_by_class['survived'][csv_name][FEATURES.index(feature) - 2] += chunk
    #                 counter += ['survived'] * len(chunk)
    #                 continue
    #             if 'sick' in csv_name:
    #                 dataset_by_class['sick'][csv_name][FEATURES.index(feature) - 2] += chunk
    #                 counter += ['sick'] * len(chunk)
    #                 continue
    #
    # print(Counter(counter))
    # dataset_by_class = {key: {csv_name: np.array(dataset_by_class[key][csv_name]).T for csv_name in dataset_by_class[key].keys()} for key in dataset_by_class.keys()}
    # test_dataset_by_class = {key: {csv_name: np.array(test_dataset_by_class[key][csv_name]).T for csv_name in test_dataset_by_class[key].keys()} for key in test_dataset_by_class.keys()}
    #
    # def create_dataset(dataset_x, dataset_y, look_back=1):
    #     dataX, dataY = [], []
    #     for i in range(len(dataset_x) - look_back + 1):
    #         a = dataset_x[i:(i + look_back)]
    #         dataX.append(a)
    #         dataY.append(dataset_y[i + look_back - 1])
    #     return dataX, dataY
    #
    # look_back = 72
    # trainX, trainY = {}, {}
    # for key in dataset_by_class.keys():
    #     trainX[key], trainY[key] = {}, {}
    #     for csv_name in dataset_by_class[key].keys():
    #         if key == 'healthy':
    #             labels_train = [[1, 0, 0, 0, 0]] * len(dataset_by_class[key][csv_name])
    #         if key == 'sick':
    #             labels_train = []
    #             for s in range(len(dataset_by_class[key][csv_name])):
    #                 if s <= 0.25 * len(dataset_by_class[key][csv_name]):
    #                     label = [0, 1, 0, 0, 0, 0]
    #                 if 0.25 * len(dataset_by_class[key][csv_name]) < s <= 0.5 * len(dataset_by_class[key][csv_name]):
    #                     label = [0, 0, 1, 0, 0, 0]
    #                 if 0.5 * len(dataset_by_class[key][csv_name]) < s <= 0.75 * len(dataset_by_class[key][csv_name]):
    #                     label = [0, 0, 0, 1, 0, 0]
    #                 if 0.75 * len(dataset_by_class[key][csv_name]) < s:
    #                     label = [0, 0, 0, 0, 1, 0]
    #                 # label = [s / len(dataset_by_class[key][csv_name])]
    #                 labels_train.append(label[:-1])
    #         if key == 'survived':
    #             labels_train = [[0, 1, 0, 0, 0]] * len(dataset_by_class[key][csv_name][:144])
    #             labels_train += [[0, 0, 1, 0, 0]] * len(dataset_by_class[key][csv_name][144:288])
    #             labels_train += [[0, 0, 0, 0, 0]] * len(dataset_by_class[key][csv_name][288:])
    #
    #         trainX[key][csv_name], trainY[key][csv_name] = create_dataset(dataset_by_class[key][csv_name], labels_train, look_back)
    #
    # testX, testY = {}, {}
    # for key in test_dataset_by_class.keys():
    #     testX[key], testY[key] = {}, {}
    #     for csv_name in test_dataset_by_class[key].keys():
    #         if key == 'healthy':
    #             # labels_test = [[1, 0, 0, 0, 0, 0]] * len(test_dataset_by_class[key][csv_name])
    #             labels_test = [[0]] * len(test_dataset_by_class[key][csv_name])
    #         if key == 'sick':
    #             labels_test = []
    #             for s in range(len(test_dataset_by_class[key][csv_name])):
    #                 if s <= 0.25 * len(test_dataset_by_class[key][csv_name]):
    #                     label = [0, 1, 0, 0, 0, 0]
    #                 if 0.25 * len(test_dataset_by_class[key][csv_name]) < s <= 0.5 * len(
    #                         test_dataset_by_class[key][csv_name]):
    #                     label = [0, 0, 1, 0, 0, 0]
    #                 if 0.5 * len(test_dataset_by_class[key][csv_name]) < s <= 0.75 * len(
    #                         test_dataset_by_class[key][csv_name]):
    #                     label = [0, 0, 0, 1, 0, 0]
    #                 if 0.75 * len(test_dataset_by_class[key][csv_name]) < s:
    #                     label = [0, 0, 0, 0, 1, 0]
    #                 # label = [s / len(test_dataset_by_class[key][csv_name])]
    #                 labels_test.append(label)
    #         if key == 'survived':
    #             labels_test = [[0, 1, 0, 0, 0, 0]] * len(test_dataset_by_class[key][csv_name][:144])
    #             labels_test += [[0, 0, 1, 0, 0, 0]] * len(test_dataset_by_class[key][csv_name][144:288])
    #             labels_test += [[0, 0, 0, 0, 0, 1]] * len(test_dataset_by_class[key][csv_name][288:])
    #
    #         testX[key][csv_name], testY[key][csv_name] = create_dataset(test_dataset_by_class[key][csv_name],
    #                                                                     labels_test, look_back)
    #
    # min_class_len = 100500
    # for key in trainX.keys():
    #     key_data = []
    #     key_labels = []
    #     for csv_name in trainX[key].keys():
    #         key_data += trainX[key][csv_name]
    #         key_labels += trainY[key][csv_name]
    #     trainX[key] = key_data
    #     trainY[key] = key_labels
    #     if 0 < len(key_data) < min_class_len:
    #         min_class_len = len(key_data)
    #
    # for key in trainX.keys():
    #     if len(trainX[key]) > 0:
    #         trainX[key] = trainX[key][::len(trainX[key]) // min_class_len]
    #         trainY[key] = trainY[key][::len(trainY[key]) // min_class_len]
    #
    #     print(len(trainX[key]), len(trainY[key]))
    #
    # for key in testX.keys():
    #     key_data = []
    #     key_labels = []
    #     for csv_name in testX[key].keys():
    #         key_data += testX[key][csv_name]
    #         key_labels += testY[key][csv_name]
    #     testX[key] = key_data
    #     testY[key] = key_labels
    #
    # dataset_x = []
    # dataset_y = []
    # test_dataset_x = []
    # test_dataset_y = []
    # min_class_len = 1005000
    # for key in trainX.keys():
    #     if len(trainX[key]) < min_class_len:
    #         min_class_len = len(trainX[key])
    #
    # for key in trainX.keys():
    #     dataset_x += trainX[key]#[:min_class_len]
    #     dataset_y += trainY[key]#[:min_class_len]
    #     test_dataset_x += testX[key]
    #     test_dataset_y += testY[key]
    #
    # testX, testY = np.array(test_dataset_x), np.array(test_dataset_y)
    #
    # c = list(zip(dataset_x, dataset_y))
    # random.shuffle(c)
    # random.shuffle(c)
    # random.shuffle(c)
    # random.shuffle(c)
    # trainX, trainY = zip(*c)
    # trainY = np.array(trainY)
    #
    # # reshape input to be [samples, time steps, features]
    # trainX = np.reshape(trainX, (len(trainX), look_back, len(FEATURES) - 2))
    # testX = np.reshape(testX, (len(testX), look_back, len(FEATURES) - 2))
    # # create and fit the LSTM network
    # # model = Sequential()
    # # model.add(LSTM(64, input_shape=(look_back, len(FEATURES) - 2)))
    # # model.add(Dropout(0.5))
    # # model.add(Dense(5, activation='softmax'))
    # # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    # # n_epoch = 1
    # # acc = 0
    # # while n_epoch <= 1:
    # # #for epoch in range(n_epoch):
    # #     history = model.fit(trainX, trainY, epochs=100, batch_size=32, validation_split=0.1)#, callbacks=[EarlyStopping(monitor='val_acc', min_delta=0.001, patience=2)])
    # #     # accr = model.evaluate(testX, testY)
    # #     # print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))
    # #     # acc = accr[1]
    # #     n_epoch += 1
    #
    # from keras.models import load_model
    # model = load_model('lstm.h5')
    # # model.save('lstm.h5')
    # # accr = model.evaluate(testX, testY)
    # # print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))
    #
    # # predictions = model.predict(testX)
    # # plt.title('Loss')
    # # plt.plot(history.history['loss'], label='train')
    # # plt.plot(history.history['val_loss'], label='test')
    # # plt.legend()
    # # plt.show()
    # #
    # # plt.title('Accuracy')
    # # plt.plot(history.history['acc'], label='train')
    # # plt.plot(history.history['val_acc'], label='test')
    # # plt.legend()
    # # plt.show()
    # #
    # # plt.title('test')
    # # temp = testX[:, :, 0]
    # # for i in range(len(temp)):
    # #     if test_dataset_y[i][0] == 1:
    # #         color = 'blue'
    # #     if test_dataset_y[i][1] == 1:
    # #         color = 'green'
    # #     if test_dataset_y[i][2] == 1:
    # #         color = 'red'
    # #     plt.plot([i], [temp[i]], 'o', color=color)
    # #
    # # for i in range(0, len(temp), 10):
    # #     print(predictions[i])
    # #     if np.argmax(predictions[i]) == 0:
    # #         color = 'blue'
    # #     if np.argmax(predictions[i]) == 1:
    # #         color = 'green'
    # #     if np.argmax(predictions[i]) == 2:
    # #         color = 'red'
    # #     plt.plot([i], [temp[i] + 2], 'o', color=color)
    # #
    # # plt.show()
    #
    # for csv_name in decompose_trend_by_csv.keys():
    #     if 'survived' in csv_name or 'sick' in csv_name:#'april_healthy_1_pc' in csv_name or 'june_healthy_2_ucsf' in csv_name or 'april_healthy_3_ucsf' in csv_name or 'july_sick_3_ucsf' in csv_name or 'july_sick_2_ucsf' in csv_name or 'june_sick_2_pc' in csv_name or 'june_sick_3_ucsf' in csv_name or 'june_sick_3_ucsf' in csv_name:
    #         test_x = [[] for feature in FEATURES[2:]]
    #         for feature in FEATURES[2:]:
    #             chunk = decompose_trend_by_csv[csv_name][feature]
    #             test_x[FEATURES.index(feature) - 2] = chunk
    #         if 'healthy' in csv_name:
    #             test_y = [[1, 0, 0, 0, 0]] * len(chunk)
    #         if 'sick' in csv_name:
    #             test_y = []
    #             for s in range(len(chunk)):
    #                 if s <= 0.25 * len(chunk):
    #                     label = [0, 1, 0, 0, 0, 0]
    #                 if 0.25 * len(chunk) < s <= 0.5 * len(chunk):
    #                     label = [0, 0, 1, 0, 0, 0]
    #                 if 0.5 * len(chunk) < s <= 0.75 * len(chunk):
    #                     label = [0, 0, 0, 1, 0, 0]
    #                 if 0.75 * len(chunk) < s:
    #                     label = [0, 0, 0, 0, 1, 0]
    #                 # label = [s / len(test_dataset_by_class[key][csv_name])]
    #                 test_y.append(label[:-1])
    #         if 'survived' in csv_name:
    #             test_y = [[0, 1, 0, 0, 0, 0]] * len(chunk[:144])
    #             test_y += [[0, 0, 1, 0, 0, 0]] * len(chunk[144:288])
    #             test_y += [[0, 0, 0, 0, 0, 1]] * len(chunk[288:])
    #
    #         test_x = np.array(test_x).T
    #         test_x, test_y = create_dataset(test_x, test_y, look_back)
    #         test_x, test_y = np.array(test_x), np.array(test_y)
    #         test_x = np.reshape(test_x, (len(test_x), look_back, len(FEATURES) - 2))
    #
    #         predictions = model.predict(test_x)
    #
    #         plt.title('test {}'.format(csv_name.split('/')[-1]))
    #         temp = test_x[:, 0, 0]
    #         # for i in range(0, len(temp), 2):
    #         #     if test_y[i][0] == 1:
    #         #         color = 'blue'
    #         #     if test_y[i][5] == 1:
    #         #         color = 'green'
    #         #     if test_y[i][1] == 1:
    #         #         color = 'yellow'
    #         #     if test_y[i][2] == 1:
    #         #         color = 'orange'
    #         #     if test_y[i][3] == 1:
    #         #         color = 'red'
    #         #     if test_y[i][4] == 1:
    #         #         color = 'brown'
    #         #     plt.plot([i], [temp[i]], 'o', color=color)
    #
    #         for i in range(0, len(temp), 2):
    #             if np.argmax(predictions[i]) == 0:
    #                 color = 'blue'
    #             if np.argmax(predictions[i]) == 5:
    #                 color = 'green'
    #             if np.argmax(predictions[i]) == 1:
    #                 color = 'yellow'
    #             if np.argmax(predictions[i]) == 2:
    #                 color = 'orange'
    #             if np.argmax(predictions[i]) == 3:
    #                 color = 'red'
    #             if np.argmax(predictions[i]) == 4:
    #                 color = 'brown'
    #
    #             plt.plot([i], [temp[i] + 0.3], 'o', color=color)
    #
    #         plt.show()
