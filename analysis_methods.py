import os

import matplotlib.cm as cm
from sklearn.decomposition import PCA

from utils import *
from config import FEATURES, KS_THRESH


class TRKIR(object):
    def __init__(self, dirnames, filters_in=[], filters_not_in=[]):
        self.filters_in = filters_in
        self.filters_not_in = filters_not_in

        self.time_avg = 144
        self.dirnames = dirnames

        # COMMON PREPROCESSING
        self.load_tables()
        self.divide_by_days()
        self.divide_by_time()
        self.avg_by_day()
        self.avg_days()
        self.avg_by_time()

    def load_tables(self):
        self.csv_names = []
        for name in self.dirnames:
            if os.path.isdir(name):
                self.csv_names += sorted([os.path.join(name, d) for d in os.listdir(name) if '.csv' in d])
            else:
                self.csv_names.append(name)

        bad_csv_names = []
        for csv_name in self.csv_names:
            for x in self.filters_in:
                if x not in csv_name:
                    bad_csv_names.append(csv_name)
            for x in self.filters_not_in:
                if x in csv_name:
                    bad_csv_names.append(csv_name)
        self.csv_names = [csv_name for csv_name in self.csv_names if csv_name not in bad_csv_names]

        self.csvs = {
            csv_name.split('/')[-1].replace('_', ' ').replace('.csv', '') : clear_from_nans(load_csv(csv_name, FEATURES))
            for csv_name in self.csv_names
        }
        self.csv_names = list(self.csvs.keys())
        for csv_name in self.csv_names:
            print(csv_name)

    def divide_by_days(self):
        self.by_day_csvs = {
            csv_name: add_mock_last_day(delete_bad_days(divide_by_days(self.csvs[csv_name], full_days=True)))
            for csv_name in self.csv_names
        }

    def divide_by_time(self):
        self.by_time_csvs = {
            csv_name: divide_days_by_time(self.by_day_csvs[csv_name], self.time_avg)
            for csv_name in self.csv_names
        }

    def avg_by_day(self):
        self.avg_by_day_csvs = {
            csv_name: average_by_days(self.by_day_csvs[csv_name])
            for csv_name in self.csv_names
        }

    def avg_days(self):
        self.avg_days_csvs = {
            csv_name: average_csv_days(self.avg_by_day_csvs[csv_name])
            for csv_name in self.csv_names
        }

    def avg_by_time(self):
        self.avg_by_time_csvs = {
            csv_name: average_days_by_time(self.by_time_csvs[csv_name])
            for csv_name in self.csv_names
        }

    def plot_csvs_with_trend(self, trend='ma', equal_healthy_days=False, save=False):
        for feature in FEATURES[2:]:
            for csv_name in self.csv_names:
                if 'sick' in csv_name:
                    try:
                        if equal_healthy_days:
                            individual = (self.avg_by_time_csvs[self.opposite_period_csv(csv_name, self.csv_names)][
                                              feature] * 6)[: 6]
                            dates = (self.avg_by_time_csvs[self.opposite_period_csv(csv_name, self.csv_names)][
                                 FEATURES[0]] * 6)[: 6]
                            times = (self.avg_by_time_csvs[self.opposite_period_csv(csv_name, self.csv_names)][
                                         FEATURES[1]] * 6)[: 6]
                        else:
                            individual = self.avg_by_time_csvs[self.opposite_period_csv(csv_name, self.csv_names)][
                                feature]
                            dates = self.avg_by_time_csvs[self.opposite_period_csv(csv_name, self.csv_names)][
                                         FEATURES[0]]
                            times = self.avg_by_time_csvs[self.opposite_period_csv(csv_name, self.csv_names)][
                                         FEATURES[1]]
                    except:
                        continue

                    individual += self.avg_by_time_csvs[csv_name][feature]
                    dates += self.avg_by_time_csvs[csv_name][FEATURES[0]]
                    times += self.avg_by_time_csvs[csv_name][FEATURES[1]]

                    individual = list(itertools.chain.from_iterable(individual))
                    dates = list(itertools.chain.from_iterable(dates))
                    times = list(itertools.chain.from_iterable(times))

                    fig = plt.figure(figsize=(14, 10))

                    csv_name = csv_name.replace('sick', '').replace('survived', '')
                    ax = fig.add_subplot(111)
                    ax.set_xlabel('x', fontsize=5)
                    ax.set_ylabel('y', fontsize=5)
                    ax.set_title(feature + ' with {} for {}'.format(trend, csv_name), fontsize=10)
                    dates_ticks = [i for i in range(0, len(dates), 144)]
                    ax.set_xticks(dates_ticks)
                    ax.set_xticklabels([dates[int(i)] + ' ' + times[int(i)] for i in dates_ticks], rotation=20)

                    ax.plot(individual, label='raw')

                    if trend == 'ma':
                        ma = moving_average(individual, window_size=100)
                        ax.plot(ma, linewidth=5, label='moving avg.')
                    if trend == 'tl':
                        ax.plot(trendline(data[i]), linewidth=5, label='trend line')

                    if save:
                        plt.savefig(os.path.join('plots', trend, '{}_{}_{}.png'.format(feature, csv_name, trend)))
                        plt.close()
                    else:
                        plt.show()

    @staticmethod
    def transform_data_for_pca(data, days=False):
        if days:
            for key in FEATURES[2:]:
                data[key] = [data[key]]

        return np.array([data[key] for key in FEATURES[2:]]).T.tolist()

    def feature_pca_by_days(self):
        pca_data = self.pca_for_csvs_by_days({
            csv_name: self.transform_data_for_pca(self.avg_by_day_csvs[csv_name])
            for csv_name in self.csv_names
        })
        pd_to_csv(self.plot_pca_for_csvs_by_days(pca_data, plot=True, save=True, title='feature/{}/pca_for_days'.format(self.filters_in[-1])), 'pca/feature/{}/pca_for_days.csv'.format(self.filters_in[-1]))

    def feature_pca_by_individuals(self):
        pca_data = self.pca_for_csvs_by_days({
            csv_name: self.transform_data_for_pca(self.avg_days_csvs[csv_name], days=True)
            for csv_name in self.csv_names
        })
        pd_to_csv(self.plot_pca_for_csvs_by_days(pca_data, plot=True, save=True, title='feature/{}/pca_for_individuals'.format(self.filters_in[-1])), 'pca/feature/{}/pca_for_individuals.csv'.format(self.filters_in[-1]))

    def create_avg_healthy_day(self):
        self.avg_healthy_day = {
            csv_name: self.create_average_healthy_day({
                feature: self.avg_by_time_csvs[csv_name][feature] for feature in FEATURES[2:]
            })
            for csv_name in self.csv_names if 'healthy' in csv_name
        }
        self.avg_healthy_day = {
            csv_name: {
                feature: self.avg_healthy_day[csv_name][feature] * 20 for feature in FEATURES[2:]
            }
            for csv_name in self.csv_names if 'healthy' in csv_name
        }

    def feature_time_warping(self, equal_healthy_days=True):
        self.time_warpings = {}
        for feature in FEATURES[2:]:
            self.time_warpings[feature] = {}
            for csv_name in self.csv_names:
                if 'sick' in csv_name:
                    try:
                        if equal_healthy_days:
                            to_compare = (self.avg_by_time_csvs[self.opposite_period_csv(csv_name, self.csv_names)][feature] * 6)[: 6]
                            dates = (self.avg_by_time_csvs[self.opposite_period_csv(csv_name, self.csv_names)][
                                         FEATURES[0]] * 6)[: 6]
                            times = (self.avg_by_time_csvs[self.opposite_period_csv(csv_name, self.csv_names)][
                                         FEATURES[1]] * 6)[: 6]
                        else:
                            to_compare = self.avg_by_time_csvs[self.opposite_period_csv(csv_name, self.csv_names)][feature]
                            dates = self.avg_by_time_csvs[self.opposite_period_csv(csv_name, self.csv_names)][
                                         FEATURES[0]]
                            times = self.avg_by_time_csvs[self.opposite_period_csv(csv_name, self.csv_names)][
                                         FEATURES[1]]
                    except:
                        continue

                    n_healthy = len(to_compare)
                    to_compare += self.avg_by_time_csvs[csv_name][feature]
                    with_compare = self.avg_healthy_day[self.opposite_period_csv(csv_name, self.csv_names)][feature][: len(to_compare) * len(to_compare[0]) + 1]

                    self.time_warpings[feature][csv_name] = {
                        'data': time_warping(to_compare, with_compare),
                        'n_healthy': n_healthy,
                    }

                    dates += self.avg_by_time_csvs[csv_name][FEATURES[0]][:]
                    times += self.avg_by_time_csvs[csv_name][FEATURES[1]][:]
                    dates = list(itertools.chain.from_iterable(dates))
                    times = list(itertools.chain.from_iterable(times))

                    # plot_feature_with_trends(
                    #     [np.array(list(itertools.chain.from_iterable(to_compare))) * 3, np.array(with_compare) * 3, self.time_warpings[feature][csv_name]['data']],
                    #     dates=[dates[i] + ' ' + times[i] for i in range(len(dates))],
                    #     trend='tw',
                    #     title='tw_{}_{}'.format(csv_name.split('/')[-1].replace('sick', ''), feature),
                    #     save=True,
                    # )

            df = pd.DataFrame.from_dict({
                csv_name: self.time_warpings[feature][csv_name]['data']
                for csv_name in self.time_warpings[feature].keys()
            }, orient='index')
            df = df.transpose()
            pd_to_csv(df, 'tw/tw_{}.csv'.format(feature))

        self.time_warpings = {
            csv_name: {
                feature: self.time_warpings[feature][csv_name] for feature in FEATURES[2:]
            }
            for csv_name in self.time_warpings[FEATURES[-1]].keys()
        }

    def time_warping_pca_by_days(self):
        tw_pca_days = {}
        for csv_name in self.time_warpings.keys():
            tw_pca_days[csv_name.replace('sick', '').replace('survived', '') + ' healthy'] = {
                feature: np.mean(np.array_split(self.time_warpings[csv_name][feature]['data'],
                                                len(self.time_warpings[csv_name][feature]['data']) // 144)[: self.time_warpings[csv_name][feature]['n_healthy']],
                                 axis=1)
                for feature in FEATURES[2:]
            }
            tw_pca_days[csv_name.replace('sick', '') + ' sick'] = {
                feature: np.mean(np.array_split(self.time_warpings[csv_name][feature]['data'],
                                                len(self.time_warpings[csv_name][feature]['data']) // 144)[self.time_warpings[csv_name][feature]['n_healthy']: ],
                                 axis=1)
                for feature in FEATURES[2:]
            }

        tw_pca_days = {
            csv_name: self.transform_data_for_pca(tw_pca_days[csv_name])
            for csv_name in tw_pca_days.keys()
        }

        pca_data = self.pca_for_csvs_by_days(tw_pca_days)
        pd_to_csv(self.plot_pca_for_csvs_by_days(pca_data, plot=True, save=True, title='tw/{}/tw_pca_for_days'.format(self.filters_in[-1])), 'pca/tw/{}/tw_pca_for_days.csv'.format(self.filters_in[-1]))

    def time_warping_pca_by_individuals(self):
        tw_pca_days = {}
        for csv_name in self.time_warpings.keys():
            tw_pca_days[csv_name.replace('sick', '').replace('survived', '') + ' healthy'] = {
                feature: np.mean(self.time_warpings[csv_name][feature]['data'][: 144 * self.time_warpings[csv_name][feature]['n_healthy']])
                for feature in FEATURES[2:]
            }
            tw_pca_days[csv_name.replace('sick', '') + ' sick'] = {
                feature: np.mean(self.time_warpings[csv_name][feature]['data'][144 * self.time_warpings[csv_name][feature]['n_healthy']:])
                for feature in FEATURES[2:]
            }

        tw_pca_days = {
            csv_name: self.transform_data_for_pca(tw_pca_days[csv_name], days=True)
            for csv_name in tw_pca_days.keys()
        }

        pca_data = self.pca_for_csvs_by_days(tw_pca_days)
        pd_to_csv(self.plot_pca_for_csvs_by_days(pca_data, plot=True, save=True, title='tw/{}/tw_pca_for_individuals'.format(self.filters_in[-1])), 'pca/tw/{}/tw_pca_for_individuals.csv'.format(self.filters_in[-1]))

    def feature_seasonal_decomposition(self, equal_healthy_days=True):
        self.seasonal_decomposition = {}
        for feature in FEATURES[2:]:
            self.seasonal_decomposition[feature] = {}
            for csv_name in self.csv_names:
                if 'sick' in csv_name:
                    try:
                        if equal_healthy_days:
                            individual = (self.avg_by_time_csvs[self.opposite_period_csv(csv_name, self.csv_names)][
                                              feature] * 6)[: 6]
                            dates = (self.avg_by_time_csvs[self.opposite_period_csv(csv_name, self.csv_names)][
                                         FEATURES[0]] * 6)[: 6]
                            times = (self.avg_by_time_csvs[self.opposite_period_csv(csv_name, self.csv_names)][
                                         FEATURES[1]] * 6)[: 6]
                        else:
                            individual = self.avg_by_time_csvs[self.opposite_period_csv(csv_name, self.csv_names)][
                                feature]
                            dates = self.avg_by_time_csvs[self.opposite_period_csv(csv_name, self.csv_names)][
                                FEATURES[0]]
                            times = self.avg_by_time_csvs[self.opposite_period_csv(csv_name, self.csv_names)][
                                FEATURES[1]]
                    except:
                        individual = []

                    individual += self.avg_by_time_csvs[csv_name][feature]

                    n_healthy = len(to_compare)
                    self.time_warpings[feature][csv_name] = {
                        'data': time_warping(to_compare, with_compare),
                        'n_healthy': n_healthy,
                    }
                    dates += self.avg_by_time_csvs[csv_name][FEATURES[0]][:]
                    times += self.avg_by_time_csvs[csv_name][FEATURES[1]][:]
                    dates = list(itertools.chain.from_iterable(dates))
                    times = list(itertools.chain.from_iterable(times))

                    self.seasonal_decomposition[feature][csv_name] = seasonal_decompose(
                        list(itertools.chain.from_iterable(individual)),
                        model='additive',
                        freq=144
                    )

                    self.seasonal_decomposition[feature][csv_name] = {
                        'trend': self.seasonal_decomposition[feature][csv_name].trend.tolist()[144: -144],
                        'seasonal': self.seasonal_decomposition[feature][csv_name].seasonal.tolist()[:144],
                        'resid': self.seasonal_decomposition[feature][csv_name].resid.tolist()[144: -144],
                        'n_healthy': n_healthy - 1
                    }

                    plot_feature_with_trends(
                        [list(itertools.chain.from_iterable(individual)), [None] * 144 + self.seasonal_decomposition[feature][csv_name]['trend']],
                        dates=[dates[i] + ' ' + times[i] for i in range(len(dates))],
                        trend='tw',
                        title='tw_{}_{}'.format(csv_name.split('/')[-1].replace('sick', ''), feature),
                        save=True,
                    )

            df = pd.DataFrame.from_dict({
                csv_name: self.seasonal_decomposition[feature][csv_name]['trend']
                for csv_name in self.seasonal_decomposition[feature].keys()
            }, orient='index')
            df = df.transpose()
            pd_to_csv(df, 'sd/trend_{}.csv'.format(feature))

            df = pd.DataFrame.from_dict({
                csv_name: self.seasonal_decomposition[feature][csv_name]['seasonal']
                for csv_name in self.seasonal_decomposition[feature].keys()
            }, orient='index')
            df = df.transpose()
            pd_to_csv(df, 'sd/seasonal_{}.csv'.format(feature))

            df = pd.DataFrame.from_dict({
                csv_name: self.seasonal_decomposition[feature][csv_name]['resid']
                for csv_name in self.seasonal_decomposition[feature].keys()
            }, orient='index')
            df = df.transpose()
            pd_to_csv(df, 'sd/resid_{}.csv'.format(feature))

        self.seasonal_decomposition = {
            csv_name: {
                feature: self.seasonal_decomposition[feature][csv_name] for feature in FEATURES[2:]
            }
            for csv_name in self.seasonal_decomposition[FEATURES[-1]].keys()
        }

    def seasonal_decomposition_pca_by_days(self):
        trend_pca_days = {
            csv_name: {
                feature: np.mean(
                    np.array_split(
                        self.seasonal_decomposition[csv_name][feature]['trend'], len(self.seasonal_decomposition[csv_name][feature]['trend']) // 144
                    ), axis=1
                )
                for feature in FEATURES[2:]
            }
            for csv_name in self.seasonal_decomposition.keys()
        }
        trend_pca_days = {
            csv_name: self.transform_data_for_pca(trend_pca_days[csv_name])
            for csv_name in trend_pca_days.keys()
        }

        pca_data = self.pca_for_csvs_by_days(trend_pca_days)
        pd_to_csv(self.plot_pca_for_csvs_by_days(pca_data, plot=True), 'pca/trend_for_days.csv')

        seasonal_pca_days = {
            csv_name: {
                feature: np.mean(
                    np.array_split(
                        self.seasonal_decomposition[csv_name][feature]['seasonal'],
                        1
                    ), axis=1
                )
                for feature in FEATURES[2:]
            }
            for csv_name in self.seasonal_decomposition.keys()
        }
        seasonal_pca_days = {
            csv_name: self.transform_data_for_pca(seasonal_pca_days[csv_name])
            for csv_name in seasonal_pca_days.keys()
        }

        pca_data = self.pca_for_csvs_by_days(seasonal_pca_days)
        pd_to_csv(self.plot_pca_for_csvs_by_days(pca_data, plot=True), 'pca/season_for_days.csv')

        resid_pca_days = {
            csv_name: {
                feature: np.mean(
                    np.array_split(
                        self.seasonal_decomposition[csv_name][feature]['resid'],
                        len(self.seasonal_decomposition[csv_name][feature]['resid']) // 144
                    ), axis=1
                )
                for feature in FEATURES[2:]
            }
            for csv_name in self.seasonal_decomposition.keys()
        }
        resid_pca_days = {
            csv_name: self.transform_data_for_pca(resid_pca_days[csv_name])
            for csv_name in resid_pca_days.keys()
        }

        pca_data = self.pca_for_csvs_by_days(resid_pca_days)
        pd_to_csv(self.plot_pca_for_csvs_by_days(pca_data, plot=True), 'pca/resid_for_days.csv')

    def seasonal_decomposition_pca_by_individuals(self):
        trend_pca_days = {
            csv_name: {
                feature: np.mean(self.seasonal_decomposition[csv_name][feature]['trend'])
                for feature in FEATURES[2:]
            }
            for csv_name in self.seasonal_decomposition.keys()
        }
        trend_pca_days = {
            csv_name: self.transform_data_for_pca(trend_pca_days[csv_name], days=True)
            for csv_name in trend_pca_days.keys()
        }

        pca_data = self.pca_for_csvs_by_days(trend_pca_days)
        pd_to_csv(self.plot_pca_for_csvs_by_days(pca_data, plot=True), 'pca/trend_for_individuals.csv')

        seasonal_pca_days = {
            csv_name: {
                feature: np.mean(self.seasonal_decomposition[csv_name][feature]['seasonal'])
                for feature in FEATURES[2:]
            }
            for csv_name in self.seasonal_decomposition.keys()
        }
        seasonal_pca_days = {
            csv_name: self.transform_data_for_pca(seasonal_pca_days[csv_name], days=True)
            for csv_name in seasonal_pca_days.keys()
        }

        pca_data = self.pca_for_csvs_by_days(seasonal_pca_days)
        pd_to_csv(self.plot_pca_for_csvs_by_days(pca_data, plot=True), 'pca/seasonal_for_individuals.csv')

        resid_pca_days = {
            csv_name: {
                feature: np.mean(self.seasonal_decomposition[csv_name][feature]['resid'])
                for feature in FEATURES[2:]
            }
            for csv_name in self.seasonal_decomposition.keys()
        }
        resid_pca_days = {
            csv_name: self.transform_data_for_pca(resid_pca_days[csv_name], days=True)
            for csv_name in resid_pca_days.keys()
        }

        pca_data = self.pca_for_csvs_by_days(resid_pca_days)
        pd_to_csv(self.plot_pca_for_csvs_by_days(pca_data, plot=True), 'pca/resid_for_individuals.csv')

    def make_individuals_ssa(self):
        self.individuals_ssa = {}
        for csv_name in self.csv_names:
            if 'sick' in csv_name:
                self.individuals_ssa[csv_name] = {}
                for feature in FEATURES[2:]:
                    n_healthy = 0
                    try:
                        healthy_individual = self.avg_by_time_csvs[self.opposite_period_csv(csv_name, self.csv_names)][
                            feature]
                        n_healthy = len(healthy_individual)
                    except:
                        del self.individuals_ssa[csv_name]
                        break
                    try:
                        sick_individual = self.avg_by_time_csvs[csv_name][feature]
                    except:
                        del self.individuals_ssa[csv_name]
                        break

                    self.individuals_ssa[csv_name][feature] = {
                        'n_healthy': n_healthy,
                        'data': SSA(list(itertools.chain.from_iterable(healthy_individual + sick_individual)), 72).TS_comps
                    }

    def ssa_pca(self):
        if not hasattr(self, 'individuals_ssa'):
            self.make_individuals_ssa()

        for feature in FEATURES[2:]:
            comp = 0
            n_comp = 10
            while comp < n_comp:
                pca_data = {}
                for csv_name in self.individuals_ssa.keys():
                    n_comp = min([self.individuals_ssa[csv_name][feature]['data'].shape[1], n_comp])
                    pca_data[csv_name.replace('sick', '').replace('survived', '') + ' healthy'] = np.array(np.array_split(self.individuals_ssa[csv_name][feature]['data'][:, comp], len(self.individuals_ssa[csv_name][feature]['data'][:, comp]) // 144)).tolist()[: self.individuals_ssa[csv_name][feature]['n_healthy']]
                    pca_data[csv_name.replace('sick', '') + ' sick'] = np.array(np.array_split(self.individuals_ssa[csv_name][feature]['data'][:, comp], len(self.individuals_ssa[csv_name][feature]['data'][:, comp]) // 144)).tolist()[self.individuals_ssa[csv_name][feature]['n_healthy']:]

                pca_data = self.pca_for_csvs_by_days(pca_data)
                pd_to_csv(self.plot_pca_for_csvs_by_days(pca_data, plot=True, save=True, title='ssa/{}/ssa_{}_{}_comp_for_days.csv'.format(self.filters_in[-1], feature, comp)), 'pca/ssa/{}/ssa_{}_{}_comp_for_days.csv'.format(self.filters_in[-1], feature, comp))

                comp += 1

    def correlation(self, data_type='raw', window=36, equal_healthy_days=False, save=False):
        for csv_name in self.csv_names:
            if 'sick' in csv_name:
                correlations = {}
                for feature_1 in FEATURES[2:]:
                    for feature_2 in FEATURES[2:]:
                        if feature_2 != feature_1 and feature_1 + '/' + feature_2 not in correlations.keys() and feature_2 + '/' + feature_1 not in correlations.keys():
                            try:
                                if equal_healthy_days:
                                    individual_1 = (self.avg_by_time_csvs[
                                                      self.opposite_period_csv(csv_name, self.csv_names)][
                                                      feature_1] * 6)[: 6]
                                    individual_2 = (self.avg_by_time_csvs[
                                                        self.opposite_period_csv(csv_name, self.csv_names)][
                                                        feature_2] * 6)[: 6]
                                    dates = (self.avg_by_time_csvs[self.opposite_period_csv(csv_name, self.csv_names)][
                                                 FEATURES[0]] * 6)[: 6]
                                    times = (self.avg_by_time_csvs[self.opposite_period_csv(csv_name, self.csv_names)][
                                                 FEATURES[1]] * 6)[: 6]
                                else:
                                    individual_1 = \
                                    self.avg_by_time_csvs[self.opposite_period_csv(csv_name, self.csv_names)][
                                        feature_1][:]
                                    individual_2 = \
                                    self.avg_by_time_csvs[self.opposite_period_csv(csv_name, self.csv_names)][
                                        feature_2][:]
                                    dates = self.avg_by_time_csvs[self.opposite_period_csv(csv_name, self.csv_names)][
                                                 FEATURES[0]][:]
                                    times = self.avg_by_time_csvs[self.opposite_period_csv(csv_name, self.csv_names)][
                                                 FEATURES[1]][:]
                            except:
                                break

                            individual_1 += self.avg_by_time_csvs[csv_name][feature_1][:]
                            individual_2 += self.avg_by_time_csvs[csv_name][feature_2][:]

                            dates += self.avg_by_time_csvs[csv_name][FEATURES[0]][:]
                            times += self.avg_by_time_csvs[csv_name][FEATURES[1]][:]

                            individual_1_raw = list(itertools.chain.from_iterable(individual_1))
                            individual_2_raw = list(itertools.chain.from_iterable(individual_2))
                            dates = list(itertools.chain.from_iterable(dates))
                            times = list(itertools.chain.from_iterable(times))

                            if data_type == 'raw':
                                individual_1 = individual_1_raw[:]
                                individual_2 = individual_2_raw[:]

                            if data_type == 'trend':
                                individual_1 = SSA(individual_1_raw, 72).TS_comps[:, 0]
                                individual_2 = SSA(individual_2_raw, 72).TS_comps[:, 0]

                            correlations[feature_1 + '/' + feature_2] = compute_correlations(individual_1, individual_2, window=window)

                            plt.figure(figsize=(14, 8))

                            plt.subplot(311)
                            plt.title(csv_name.split('/')[-1].replace('sick', '').replace('survived', '') + ' ' + feature_1 + '/' + feature_2 + ' correlation')
                            plt.plot(individual_1_raw, linewidth=1, label=feature_1 + ' raw')
                            plt.plot(individual_1, linewidth=3, label=feature_1 + ' trend')
                            plt.legend()

                            plt.subplot(312)
                            plt.plot(individual_2_raw, linewidth=1, label=feature_2 + ' raw')
                            plt.plot(individual_2, linewidth=3, label=feature_2 + ' trend')
                            plt.legend()

                            plt.subplot(313)
                            plt.plot(correlations[feature_1 + '/' + feature_2], label='pearson\'s coef. by {}hr window'.format(window // 6))
                            plt.plot(trendline(correlations[feature_1 + '/' + feature_2]), linewidth=3, label='trend line')
                            dates_ticks = [i for i in range(0, len(dates), 144)]
                            plt.xticks(dates_ticks, [dates[int(i)] + ' ' + times[int(i)] for i in dates_ticks], rotation=20)
                            plt.legend()

                            if save:
                                plt.savefig(os.path.join('correlation', csv_name.split('/')[-1].replace('sick', '').replace('survived', '') + '_' + feature_1 + '_' + feature_2 + '.png'))
                                plt.close()
                            else:
                                plt.show()

    def dtw_search_similar(self, csv_name_to_search, period, thresh={'healthy': 0.5, 'sick': 0.5}):
        self.individuals = {}

        if not hasattr(self, 'dtw_search'):
            self.dtw_search = {}
        self.dtw_search[csv_name_to_search] = {}

        for csv_name in self.csv_names:
            if 'sick' in csv_name:
                self.individuals[csv_name] = {}
                for feature in FEATURES[2:]:
                    try:
                        healthy_individual = (self.avg_by_time_csvs[self.opposite_period_csv(csv_name, self.csv_names)][
                            feature] * 6)[:6]

                        # TODO: only sick
                        healthy_individual = []

                        full_individual = healthy_individual[:]
                        healthy_individual = [healthy_individual[i] for i in period['healthy']]
                    except:
                        full_individual = []
                        healthy_individual = []
                        del self.individuals[csv_name]
                        break
                    try:
                        sick_individual = self.avg_by_time_csvs[csv_name][feature][:]
                        full_individual += sick_individual[:]
                        sick_individual = [sick_individual[i] for i in period['sick']]
                    except:
                        del self.individuals[csv_name]
                        break

                    self.individuals[csv_name][feature] = {
                        'full': SSA(list(itertools.chain.from_iterable(full_individual)), 72).TS_comps[:, 0],
                        'search': {
                            # 'healthy': SSA(list(itertools.chain.from_iterable(healthy_individual)), 72).TS_comps[:, 0],
                            'sick': SSA(list(itertools.chain.from_iterable(sick_individual)), 72).TS_comps[:, 0],
                        }
                    }

        for csv_name in self.individuals.keys():
            if csv_name != csv_name_to_search:
                self.dtw_search[csv_name_to_search][csv_name] = {}
                for feature in FEATURES[2:]:
                    self.dtw_search[csv_name_to_search][csv_name][feature] = {}

                    if period['healthy']:
                        self.dtw_search[csv_name_to_search][csv_name][feature]['healthy'] = mine_dtw(
                            self.individuals[csv_name][feature]['search']['healthy'],
                            self.individuals[csv_name_to_search][feature]['search']['healthy']
                        )
                    if period['sick']:
                        self.dtw_search[csv_name_to_search][csv_name][feature]['sick'] = mine_dtw(
                            self.individuals[csv_name][feature]['search']['sick'],
                            self.individuals[csv_name_to_search][feature]['search']['sick']
                        )

        max_dtw = {}
        if period['healthy']:
            max_dtw['healthy'] = {
                feature: np.max([self.dtw_search[csv_name_to_search][csv_name][feature]['healthy'] for csv_name in
                                 self.dtw_search[csv_name_to_search].keys()])
                for feature in FEATURES[2:]
            }
        if period['sick']:
            max_dtw['sick'] = {
                feature: np.max([self.dtw_search[csv_name_to_search][csv_name][feature]['sick'] for csv_name in
                                 self.dtw_search[csv_name_to_search].keys()])
                for feature in FEATURES[2:]
            }

        for csv_name in self.dtw_search[csv_name_to_search].keys():
            for feature in FEATURES[2:]:
                if period['healthy']:
                    self.dtw_search[csv_name_to_search][csv_name][feature]['healthy'] = 1 - self.dtw_search[csv_name_to_search][csv_name][feature]['healthy'] / max_dtw['healthy'][feature]
                if period['sick']:
                    self.dtw_search[csv_name_to_search][csv_name][feature]['sick'] = 1 - self.dtw_search[csv_name_to_search][csv_name][feature]['sick'] / max_dtw['sick'][feature]

        similar = {}
        display_feature = 'temp'
        for csv_name in self.dtw_search[csv_name_to_search].keys():
            is_good = True
            for feature in [display_feature]:#FEATURES[2:]:
                if period['healthy'] and self.dtw_search[csv_name_to_search][csv_name][feature]['healthy'] < thresh['healthy']:
                    is_good = False
                    break
                if period['sick'] and self.dtw_search[csv_name_to_search][csv_name][feature]['sick'] < thresh['sick']:
                    is_good = False
                    break
            if is_good:
                similar[csv_name] = self.individuals[csv_name][display_feature]

        similar_individuals = list(zip(
            [csv_name for csv_name in similar.keys()],
            [similar[csv_name]['full'] for csv_name in similar.keys()],
            [self.dtw_search[csv_name_to_search][csv_name][display_feature]['healthy'] if period['healthy'] else 0 + self.dtw_search[csv_name_to_search][csv_name][display_feature]['sick'] if period['sick'] else 0 for csv_name in similar.keys()],
        ))
        similar_individuals.sort(key=lambda x: -x[-1])

        similar_names = [similar_ind[0] for similar_ind in similar_individuals]
        similar_individuals = [similar_ind[1] for similar_ind in similar_individuals]

        print(similar_names)
        return plot_one_subplot(
            [self.individuals[csv_name_to_search][display_feature]['full']] + similar_individuals,
            ['({}) similar to {}'.format(display_feature, csv_name_to_search)] + ['H score: {:.2f}, S score {:.2f} '.format(self.dtw_search[csv_name_to_search][csv_name][display_feature]['healthy'] if period['healthy'] else 1, self.dtw_search[csv_name_to_search][csv_name][display_feature]['sick'] if period['sick'] else 1) + csv_name for csv_name in similar_names],
            save=True,
            save_name='{}_{}_days'.format(csv_name_to_search, ''.join([str(i + 1) for i in range(len(period['sick']))]))
        )

    @staticmethod
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

    @staticmethod
    def plot_pca_for_csvs_by_days(pca_full_data, plot=True, save=False, title=None):
        pca_data = pca_full_data['pca']

        csv_names = list(pca_data.keys())
        pc_names = ['PC_{}'.format(i + 1) + ' {0:.2f}'.format(pca_full_data['evr'][i]) for i in
                    range(len(pca_full_data['evr']))]

        concatenated_pca = []
        targets = []
        for csv_name in csv_names:
            concatenated_pca += pca_data[csv_name]
            if len(pca_data[csv_name]) > 1:
                targets += [csv_name + ' ' + str(day) for day in
                            range(len(pca_data[csv_name]))]
            else:
                #   WITHOUT DAYS
                targets += [csv_name for day in range(len(pca_data[csv_name]))]
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

            for target, color in zip(targets, colors):

                # TODO: SH*T!!!
                if 'healthy' in target and 'april' in target:
                    color = 'blue'
                if 'sick' in target and 'april' in target:
                    color = 'darkblue'
                if 'survived' in target and 'april' in target:
                    color = 'lightblue'
                if 'healthy' in target and 'june' in target:
                    color = 'green'
                if 'sick' in target and 'june' in target:
                    color = 'darkgreen'
                if 'survived' in target and 'june' in target:
                    color = 'lightgreen'
                if 'healthy' in target and 'july' in target:
                    color = 'red'
                if 'sick' in target and 'july' in target:
                    color = 'darkred'
                if 'survived' in target and 'july' in target:
                    color = 'orange'
                if 'healthy' in target and 'august' in target:
                    color = 'gray'
                if 'sick' in target and 'august' in target:
                    color = 'black'
                if 'survived' in target and 'august' in target:
                    color = 'lightgrey'


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

            if save:
                plt.savefig(os.path.join('pca/plots', title + '.png'))
                plt.close()
            else:
                plt.show()

        return finalDf

    @staticmethod
    def create_average_healthy_day(data_by_days, days=None):
        if days:
            return {key: np.mean([data_by_days[key][day] for day in days], axis=0).tolist() for key in FEATURES[2:]}

        return {key: np.mean(data_by_days[key], axis=0).tolist() for key in FEATURES[2:]}

    @staticmethod
    def opposite_period_csv(csv_name, csv_names):
        if 'healthy' in csv_name:
            name_sick = csv_name.replace('healthy', 'sick')
            name_surv = name_sick + ' survived'

            for c_n in csv_names:
                if c_n in [name_sick, name_surv]:
                    return c_n
        else:
            if 'sick' in csv_name:
                name_healthy = csv_name.replace('sick', 'healthy')
            if ' survived' in csv_name:
                name_healthy = name_healthy.replace(' survived', '')

            for c_n in csv_names:
                if c_n == name_healthy:
                    return c_n


if __name__ == '__main__':
    csv_dir_names = sys.argv[1:]
    analysis = TRKIR(csv_dir_names, filters_in=['april'], filters_not_in=[])
    ## SHOW
    # print('SHOW')
    # analysis.plot_csvs_with_trend(save=True, equal_healthy_days=True)
    ### ANALYSIS
    ## PCA
    # print('PCA')
    # analysis.feature_pca_by_days()
    # analysis.feature_pca_by_individuals()
    ## TIME WARPING
    # print('TIME WARPING')
    # analysis.create_avg_healthy_day()
    # analysis.feature_time_warping()
    # analysis.time_warping_pca_by_days()
    # analysis.time_warping_pca_by_individuals()
    ## SEASONAL DECOMPOSITION ANALYSIS
    print('SEASONAL DECOMPOSITION')
    analysis.feature_seasonal_decomposition()
    analysis.seasonal_decomposition_pca_by_days()
    analysis.seasonal_decomposition_pca_by_individuals()

    ## DTW SEARCH
    # print('DTW SEARCH')
    # for csv_name in analysis.csv_names:
    #     if 'sick' in csv_name:
    #         for day in range(6):
    #             print(csv_name, day)
    #             try:
    #                 analysis.dtw_search_similar(
    #                     csv_name_to_search=csv_name,
    #                     period={'healthy': [], 'sick': [i for i in range(day + 1)]},
    #                     thresh={'healthy': 0.5, 'sick': 0.5}
    #                 )
    #             except:
    #                 import traceback
    #                 traceback.print_exc()
    #                 break

    # SSA
    # print('SSA')
    # analysis.ssa_pca()

    ## CORRELATION
    # print('CORRELATION')
    # analysis.correlation(data_type='trend', equal_healthy_days=True, save=True)



