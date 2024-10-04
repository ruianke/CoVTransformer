# -*- coding: utf-8 -*-

import random, math
import os
import numpy as np
from datetime import datetime
from collections import defaultdict
import datetime
import csv
import pandas as pd

from numpy import genfromtxt
from scipy.interpolate import UnivariateSpline
import warnings

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
import copy
# import h5py
import pickle
import gzip


def load_data(path='related_files/timedata-GBR-USA.csv', stop=None):
    dataset = defaultdict(list)

    my_data = genfromtxt(path, delimiter=',')[1:, 3:]
    print(my_data.shape)

    csv.field_size_limit(100000000)
    csvFile = open(path, "r", encoding='utf-8')
    reader = csv.DictReader(csvFile)

    current = None
    start = None
    end = None
    start_date_collection = {}
    end_date_collection = {}
    num = 0

    def default_array():
        return np.array([0, 1])
    country_token = defaultdict(default_array)
    country_token['GBR'] = np.array([0, 1])
    country_token['USA'] = np.array([1, 0])

    for i, item in enumerate(reader):
        str_date = item['Day'].replace('/', '-')

        if item['Country'] + '_' + item['Pango'] != current:
            end_date_collection[current] = end
            current = item['Country'] + '_' + item['Pango']
            start_date_collection[item['Country'] + '_' + item['Pango']] = str_date
            start = datetime.datetime(int(str_date.split('-')[0]), int(str_date.split('-')[1]),
                                      int(str_date.split('-')[2]))

        date = (datetime.datetime(int(str_date.split('-')[0]), int(str_date.split('-')[1]),
                                  int(str_date.split('-')[2])) - start).days
        date = np.array([date])
        country = np.array(country_token[item['Country']])

        if stop and (stop - datetime.datetime(int(str_date.split('-')[0]), int(str_date.split('-')[1]),
                                              int(str_date.split('-')[2]))).days < 0:
            continue
        else:
            dataset[item['Country'] + '_' + item['Pango']].append(np.concatenate((date, my_data[i], country)))
            end = str_date
            num += 1

    csvFile.close()
    end_date_collection[current] = end

    for key, value in dataset.items():
        dataset[key] = np.stack(value, axis=0)

    print(num)

    return dataset, start_date_collection, end_date_collection


def load_data_new(path, country, suffix='', consecutive_days=2):
    def default_array():
        return np.array([0, 1])
    country_token = defaultdict(default_array)
    country_token['GBR'] = np.array([0, 1])
    country_token['USA'] = np.array([1, 0])

    file_path = os.path.join(path, 'freqs_bySubDate_' + country + suffix + '.csv')

    df = pd.read_csv(file_path)

    # Find start date
    start_date = df['CollectionDate'].min()
    print(start_date)

    real_start_date = '2020-01-01'
    start = datetime.datetime(int(real_start_date.split('-')[0]), int(real_start_date.split('-')[1]),
                              int(real_start_date.split('-')[2]))

    def deal_str_date(str_date, start=start):
        days = (datetime.datetime(int(str_date.split('-')[0]), int(str_date.split('-')[1]),
                                  int(str_date.split('-')[2])) - start).days
        return days

    # Sort the data by 'Pango', 'AccessDate', 'CollectionDate'
    df_sort = df.sort_values(['Pango', 'AccessDate', 'CollectionDate'])[
        ['Pango', 'AccessDate', 'CollectionDate', 'totalSeqs', 'VarSeqs', 'VarFreq']]

    # Change date to abosulte value
    df_sort['AccessDay'] = df_sort['AccessDate'].apply(deal_str_date)
    df_sort['CollectionDay'] = df_sort['CollectionDate'].apply(deal_str_date)

    df_sort = df_sort[
        ['Pango', 'AccessDate', 'AccessDay', 'CollectionDate', 'CollectionDay', 'totalSeqs', 'VarSeqs', 'VarFreq']]

    result_dict = first_process_data(df_sort, country, country_token, consecutive_days, start)

    return start_date, result_dict, df_sort


def load_label(path, country):
    file_path = os.path.join(path, 'freqs_final_' + country + '.csv')

    df = pd.read_csv(file_path)

    # Find start date
    start_date = df['CollectionDate'].min()
    print(start_date)

    real_start_date = '2020-01-01'
    start = datetime.datetime(int(real_start_date.split('-')[0]), int(real_start_date.split('-')[1]),
                              int(real_start_date.split('-')[2]))

    def deal_str_date(str_date, start=start):
        days = (datetime.datetime(int(str_date.split('-')[0]), int(str_date.split('-')[1]),
                                  int(str_date.split('-')[2])) - start).days
        return days

    # Sort the data by 'Pango', 'AccessDate', 'CollectionDate'
    df_sort = df.sort_values(['Pango', 'CollectionDate'])[
        ['Pango', 'CollectionDate', 'totalSeqs', 'VarSeqs', 'VarFreq']]

    # Change date to abosulte value
    df_sort['CollectionDay'] = df_sort['CollectionDate'].apply(deal_str_date)

    df_sort = df_sort[['Pango', 'CollectionDate', 'CollectionDay', 'totalSeqs', 'VarSeqs', 'VarFreq']]

    # Generate a dictionary with 'Pango' as keys and all other columns as values
    result_dict = {}
    missing_list = set()
    for pango, group in df_sort.groupby('Pango'):
        data = group.values.tolist()

        # Loop over the sublists to find the first sublist with consecutive dates
        # missing = False
        # for i, sublist in enumerate(data):
        #     if i and sublist[2] != data[i-1][2] + 1:
        #         missing_list.append(pango)
        #         missing = True
        #         break
        # if not missing:
        #     data = [x[2:] for x in data]
        # else:
        #   continue

        data = [x[2:] for x in data]

        prev_date = None
        new_data = []

        for i, sublist in enumerate(data):
            date = sublist[0]

            if prev_date is not None and date > prev_date + 1:
                missing_list.add(pango)
                gap = date - prev_date - 1
                new_lol = []
                for i in range(prev_date + 1, date):
                    new_list = [i] + [np.nan] * (len(sublist) - 1)
                    new_lol.append(new_list)
                new_data += new_lol

            new_data.append(sublist)
            prev_date = date

        result_dict[country + '_' + pango] = np.array(new_data)

    print(country, f' Label Has {int(len(missing_list))} Missing:')
    # print(country, f' Label Has {int(len(missing_list))} Missing:', sorted(list(missing_list)))

    return start_date, result_dict, df_sort


def print_clean_dataset(dataset, start_date_collection, end_date_collection, dict_var, interp=True, token=-100):
    result = []
    clean_data = {}
    duration_dict = {}

    for key, value in dataset.items():
        dataset[key] = np.stack(value, axis=0)

        str_date1 = start_date_collection[key]
        start_day = str_date1.split('-')[0] + '-' + str_date1.split('-')[1] + '-' + str_date1.split('-')[2]

        str_date2 = end_date_collection[key]
        end_day = str_date2.split('-')[0] + '-' + str_date2.split('-')[1] + '-' + str_date2.split('-')[2]

        d1 = datetime.datetime(int(str_date1.split('-')[0]), int(str_date1.split('-')[1]), int(str_date1.split('-')[2]))
        d2 = datetime.datetime(int(str_date2.split('-')[0]), int(str_date2.split('-')[1]), int(str_date2.split('-')[2]))
        duration = (d2 - d1).days + 1

        if key in dict_var.keys():
            result.append([key, dataset[key].shape[0], start_day, end_day, duration, dict_var[key]])
        else:
            result.append([key, dataset[key].shape[0], start_day, end_day, duration, -1])

        duration_dict[key] = duration

        clean_data[key] = [[] for _ in range(duration)]
        for i in range(len(value)):
            clean_data[key][int(value[i][0])].append(value[i])

    result.sort(key=lambda x: x[-1])

    print(len(result))
    tplt = "{0:<10}\t{1:>10}\t{2:>10}\t{3:>10}\t{4:>10}\t{5:>15}"
    print(tplt.format('Outvar', 'Number', 'Start Date', 'End Date', 'Duraion Date', 'Concern Score'))
    tplt = "{0:<10}\t{1:>10}\t{2:>10}\t{3:>10}\t{4:>10}\t{5:>15.3f}"
    for r in result:
        print(tplt.format(r[0], r[1], r[2], r[3], r[4], r[5]))

    mean_dataset = defaultdict(list)
    for key, value in clean_data.items():
        for i in range(len(value)):
            data_current = np.concatenate([np.stack(value[i], axis=0).mean(axis=0)], axis=-1)
            # ,np.stack(value[i], axis=0).var(axis=0)[1:], np.stack(value[i], axis=0).max(axis=0)[1:]
            mean_dataset[key].append(data_current)

    drop_list = []
    for key, value in mean_dataset.items():
        data_current = np.stack(value, axis=0)

        # remove nan data above first one
        if interp:
            ind = np.all(np.logical_not(np.isnan(data_current)), axis=-1).argmax()
            data_current = data_current[ind:]

        flag_skip = False
        for i in range(1, data_current.shape[-1]):
            y = data_current[:, i]
            nans, x = nan_helper(y)
            if interp:
                if (nans == True).all():
                    flag_skip = True
                    break
                y[nans] = np.interp(x(nans), x(~nans), y[~nans])
            else:
                y[nans] = token
            data_current[:, i] = y
        if flag_skip:
            print(f'No value in {key}')
            drop_list.append(key)
            continue
        elif 'BA.4' in key or 'BA.5' in key:
            print(f'{key} droped')
            drop_list.append(key)
            continue
        mean_dataset[key] = data_current

    for key in drop_list:
        mean_dataset.pop(key)

    clean_dataset = defaultdict(list)

    return clean_dataset, mean_dataset, duration_dict, drop_list


def generate_cross_validation_one_country(dataset, dict_var, duration_dict, date_collection, sbar, day, input_feature,
                                          out_feature,
                                          days=5, step=5,
                                          binary=False, country='GBR', ratio=5, position=False, random_split=True,
                                          future=14, features=None, max_day=365):
    assert future >= 0
    print(country)
    train_set = [[] for _ in range(ratio)]
    train_label = [[] for _ in range(ratio)]

    exp_set, exp_label = [[] for _ in range(ratio)], [[] for _ in range(ratio)]

    linage = [[] for _ in range(ratio)]

    date_sort = []
    start_point = datetime.datetime(2020, 1, 1)

    for key, date in date_collection.items():
        if country in key:
            value = dataset[key]

            start_day = datetime.datetime(int(date.split('-')[0]), int(date.split('-')[1]), int(date.split('-')[2]))
            time = (start_day - start_point).days
            if type(value) == list or value.shape[0] < days:
                print(key, f' does not have {days} days.', end='\t')
                continue
            else:
                date_sort.append([key, time])
    date_sort.sort(key=lambda x: x[-1])
    if random_split:
        random.shuffle(date_sort)

    class_num = {0: 0, 1: 0, 2: 0}
    all = 0
    for key_day in date_sort:
        key = key_day[0]

        cs = dict_var[key]
        if cs <= sbar[0]:
            s = 0
        elif sbar[0] < cs <= sbar[1]:
            s = 1 if not binary else 0
        else:
            s = 2

        class_num[s] += 1
        all += 1

    k = [0, 0, 0]
    train_num = [{0: 0, 1: 0, 2: 0} for _ in range(ratio)]
    for key_day in date_sort:
        key = key_day[0]
        value = dataset[key]

        start_date = date_collection[key]
        start_day = datetime.datetime(int(start_date.split('-')[0]), int(start_date.split('-')[1]),
                                      int(start_date.split('-')[2]))
        time = (start_day - start_point).days

        l = (value.shape[0] - days) // step + 1

        if (value.shape[0] - days) % step:
            print(key, f': {(value.shape[0] - days) % step} days dropped.', end='\t')

        cs = dict_var[key]
        if cs <= sbar[0]:
            s = 0
        elif sbar[0] < cs <= sbar[1]:
            s = 1 if not binary else 0
        else:
            s = 2

        for i in range(l):
            # if np.max(value[i*step:i *step+day, 1]) >= 0.5:
            #   continue
            # if i*step <= 5:
            #   continue
            if i * step + day + future >= value.shape[0] or i * step + day > max_day:
                break
            if position:
                p = (np.amax(value[:i * step + day, 1]) - value[0, 1]) * np.ones([day, 1])
                if features:
                    train_set[k[s]].append(np.concatenate([value[i * step:i * step + day, features], p], axis=1))
                else:
                    train_set[k[s]].append(np.concatenate([value[i * step:i * step + day, :], p], axis=1))
            else:
                if features:
                    train_set[k[s]].append(
                        value[i * step:i * step + day, features])  # Pos_embed, days, freq, PC1, PC1 Rank, PC2
                else:
                    train_set[k[s]].append(value[i * step:i * step + day, 1:])
            linage[k[s]].append((key, s))
            if future:
                train_label[k[s]].append(value[i * step + day + future, out_feature])
            else:
                train_label[k[s]].append(np.array(cs))

            exp_set[k[s]].append(value[int(np.max([0, i * step + day - 14])):i * step + day, input_feature])
            exp_label[k[s]].append(value[i * step + day + future, out_feature])

        train_num[k[s]][s] += 1

        if train_num[k[s]][s] == round(class_num[s] / ratio):
            k[s] = k[s] + 1 if k[s] + 1 < ratio else k[s]

    print()
    print(class_num)
    print(train_num)

    for i in range(ratio):
        train_set[i] = np.stack(train_set[i], axis=0).reshape([len(train_set[i]), -1])
        train_label[i] = np.stack(train_label[i], axis=0)
        exp_label[i] = np.stack(exp_label[i], axis=0)
        print(train_set[i].shape, train_label[i].shape)

    flat_exp_set = []
    for l in exp_set:
        flat_exp_set += l

    return train_set, train_label, linage, flat_exp_set, np.concatenate(exp_label, axis=0)


def print_clean_dataset2(dataset, start_date_collection, end_date_collection, dict_var, interp=True, token=-100):
    result = []
    clean_data = {}
    duration_dict = {}

    for key, value in dataset.items():
        dataset[key] = np.stack(value, axis=0)

        str_date1 = start_date_collection[key]
        start_day = str_date1.split('-')[0] + '-' + str_date1.split('-')[1] + '-' + str_date1.split('-')[2]

        str_date2 = end_date_collection[key]
        end_day = str_date2.split('-')[0] + '-' + str_date2.split('-')[1] + '-' + str_date2.split('-')[2]

        d1 = datetime.datetime(int(str_date1.split('-')[0]), int(str_date1.split('-')[1]), int(str_date1.split('-')[2]))
        d2 = datetime.datetime(int(str_date2.split('-')[0]), int(str_date2.split('-')[1]), int(str_date2.split('-')[2]))
        duration = (d2 - d1).days + 1

        if key in dict_var.keys():
            result.append([key, dataset[key].shape[0], start_day, end_day, duration, dict_var[key]])
        else:
            result.append([key, dataset[key].shape[0], start_day, end_day, duration, -1])

        duration_dict[key] = duration

        clean_data[key] = [[] for _ in range(duration)]
        for i in range(len(value)):
            clean_data[key][int(value[i][0])].append(value[i])

    result.sort(key=lambda x: x[-1])

    print(len(result))
    tplt = "{0:<10}\t{1:>10}\t{2:>10}\t{3:>10}\t{4:>10}\t{5:>15}"
    print(tplt.format('Outvar', 'Number', 'Start Date', 'End Date', 'Duraion Date', 'Concern Score'))
    tplt = "{0:<10}\t{1:>10}\t{2:>10}\t{3:>10}\t{4:>10}\t{5:>15.3f}"
    for r in result:
        print(tplt.format(r[0], r[1], r[2], r[3], r[4], r[5]))

    mean_dataset = defaultdict(list)
    for key, value in clean_data.items():
        for i in range(len(value)):
            data_current = np.concatenate([np.stack(value[i], axis=0).mean(axis=0)], axis=-1)
            # ,np.stack(value[i], axis=0).var(axis=0)[1:], np.stack(value[i], axis=0).max(axis=0)[1:]
            mean_dataset[key].append(data_current)

    drop_list = []
    for key, value in mean_dataset.items():
        data_current = np.stack(value, axis=0)

        if 'BA.4' in key or 'BA.5' in key:
            print(f'{key} droped')
            drop_list.append(key)
            continue
        mean_dataset[key] = data_current

    for key in drop_list:
        mean_dataset.pop(key)

    clean_dataset = defaultdict(list)

    return clean_dataset, mean_dataset, duration_dict, drop_list


def generate_cross_validation_one_country2(dataset, dict_var, duration_dict, date_collection, sbar, day, input_feature,
                                           out_feature, days=5, step=5,
                                           binary=False, country='GBR', ratio=5, position=False, random_split=True,
                                           future=14, features=None, max_day=365):
    assert future >= 0
    print(country)
    train_set = [[] for _ in range(ratio)]
    train_label = [[] for _ in range(ratio)]

    exp_set, exp_label = [[] for _ in range(ratio)], [[] for _ in range(ratio)]

    linage = [[] for _ in range(ratio)]

    date_sort = []
    start_point = datetime.datetime(2020, 1, 1)

    for key, date in date_collection.items():
        if country in key:
            value = dataset[key]

            start_day = datetime.datetime(int(date.split('-')[0]), int(date.split('-')[1]), int(date.split('-')[2]))
            time = (start_day - start_point).days
            if type(value) == list or value.shape[0] < days:
                print(key, f' does not have {days} days.', end='\t')
                continue
            else:
                date_sort.append([key, time])
    date_sort.sort(key=lambda x: x[-1])
    if random_split:
        random.shuffle(date_sort)

    class_num = {0: 0, 1: 0, 2: 0}
    all = 0
    for key_day in date_sort:
        key = key_day[0]

        cs = dict_var[key]
        if cs <= sbar[0]:
            s = 0
        elif sbar[0] < cs <= sbar[1]:
            s = 1 if not binary else 0
        else:
            s = 2

        class_num[s] += 1
        all += 1

    k = [0, 0, 0]
    train_num = [{0: 0, 1: 0, 2: 0} for _ in range(ratio)]
    for key_day in date_sort:
        key = key_day[0]
        value = dataset[key]

        ind = np.all(np.logical_not(np.isnan(value)), axis=-1).argmax()

        start_date = date_collection[key]
        start_day = datetime.datetime(int(start_date.split('-')[0]), int(start_date.split('-')[1]),
                                      int(start_date.split('-')[2]))
        time = (start_day - start_point).days

        l = (value.shape[0] - days) // step + 1

        if (value.shape[0] - days) % step:
            print(key, f': {(value.shape[0] - days) % step} days dropped.', end='\t')

        cs = dict_var[key]
        if cs <= sbar[0]:
            s = 0
        elif sbar[0] < cs <= sbar[1]:
            s = 1 if not binary else 0
        else:
            s = 2

        for i in range(ind, l):
            # if np.max(value[i*step:i *step+day, 1]) >= 0.5:
            #   continue
            # if i*step <= 5:
            #   continue
            if i * step + day + future >= value.shape[0] or i * step + day > max_day + ind:
                break
            if position:
                p = (np.amax(value[:i * step + day, 1]) - value[0, 1]) * np.ones([day, 1])
                if features:
                    train_set[k[s]].append(np.concatenate([value[i * step:i * step + day, features], p], axis=1))
                else:
                    train_set[k[s]].append(np.concatenate([value[i * step:i * step + day, :], p], axis=1))
            else:
                if features:
                    train_set[k[s]].append(
                        value[i * step:i * step + day, features])  # Pos_embed, days, freq, PC1, PC1 Rank, PC2
                else:
                    train_set[k[s]].append(value[i * step:i * step + day, 1:])
            linage[k[s]].append((key, s))
            if future:
                train_label[k[s]].append(value[i * step + day + future, out_feature])
            else:
                train_label[k[s]].append(np.array(cs))

            y = value[int(np.max([0, i * step + day - 14])):i * step + day, input_feature]
            nans, x = nan_helper(y)
            if (nans == True).any():
                break
            y = value[i * step + day + future, out_feature]
            nans, x = nan_helper(y)
            if (nans == True).any():
                break

            exp_set[k[s]].append(value[int(np.max([0, i * step + day - 14])):i * step + day, input_feature])
            exp_label[k[s]].append(value[i * step + day + future, out_feature])

        train_num[k[s]][s] += 1

        if train_num[k[s]][s] == round(class_num[s] / ratio):
            k[s] = k[s] + 1 if k[s] + 1 < ratio else k[s]

    print()
    print(class_num)
    print(train_num)

    for i in range(ratio):
        train_set[i] = np.stack(train_set[i], axis=0).reshape([len(train_set[i]), -1])
        train_label[i] = np.stack(train_label[i], axis=0)
        exp_label[i] = np.stack(exp_label[i], axis=0)
        print(train_set[i].shape, train_label[i].shape)

    flat_exp_set = []
    flat_linage = []
    for l in exp_set:
        flat_exp_set += l
    for l in linage:
        flat_linage += l

    return train_set, train_label, flat_linage, flat_exp_set, np.concatenate(exp_label, axis=0)


def generate_cross_validation_one_country_exp(dataset, label, dict_var, sbar, max_day=365, skip_days=5, exp_days=14,
                                              future=14, ratio=5, position=False,
                                              features=None, input_feature=3, out_feature=3, random_split=True,
                                              token=-4, prefix_padding=True):
    assert future >= 0

    train_set = [[] for _ in range(ratio)]
    train_label = [[] for _ in range(ratio)]

    exp_set, exp_label = [[] for _ in range(ratio)], [[] for _ in range(ratio)]

    linage = [[] for _ in range(ratio)]

    date_sort = []

    for key, date in dataset.items():
        min_access = min(date.keys())
        date_sort.append([key, min_access])

    date_sort.sort(key=lambda x: x[-1][-1])
    if random_split:
        random.shuffle(date_sort)

    class_num = {0: 0, 1: 0, 2: 0, 3: 0}  # 3 is tha pango not has concern score
    all = 0
    for key_day in date_sort:
        key = key_day[0]

        if key in dict_var.keys():
            cs = dict_var[key]
            if cs <= sbar[0]:
                s = 0
            elif sbar[0] < cs <= sbar[1]:
                s = 1
            else:
                s = 2
        else:
            s = 3
        class_num[s] += 1
        all += 1

    k = [0, 0, 0, 0]
    train_num = [{0: 0, 1: 0, 2: 0, 3: 0} for _ in range(ratio)]
    no_label = []
    multi_label = []
    samll_label = []
    less_label = []
    lag_label = []

    for key_day in date_sort:
        key = key_day[0]
        access_date = dataset[key]

        if key not in label.keys():
            no_label.append(key)
            break

        if key in dict_var.keys():
            cs = dict_var[key]
            if cs <= sbar[0]:
                s = 0
            elif sbar[0] < cs <= sbar[1]:
                s = 1
            else:
                s = 2
        else:
            s = 3

        min_access = min(access_date.keys())

        for access_days, data in access_date.items():

            relevant_day = access_days[1] - min_access[1] + 1

            if relevant_day > max_day:
                break

            label_pango = label[key]
            window_end = access_days[1]
            ind_future = window_end + future

            l = label_pango[label_pango[:, 0] == ind_future]

            if l.shape[0] > 1:
                multi_label.append(key)
            elif l.shape[0] == 0:
                samll_label.append((key, access_days))
                continue

            lag_day = int(access_days[1] - data[-1, 0])
            record_day = int(exp_days - lag_day)

            if lag_day > (exp_days / 2):
                lag_label.append((key, access_days[0]))
                continue

            start = int(np.max([0, data.shape[0] - exp_days]))
            end = int(np.min([data.shape[0], access_days[1] - skip_days + 1]))

            if end - start < exp_days:
                less_label.append((key, access_days[0]))

                if not prefix_padding:
                    continue

            exp_set[k[s]].append(data[start:end, input_feature])
            exp_label[k[s]].append(l[0, out_feature])

            linage[k[s]].append((key, s))

        train_num[k[s]][s] += 1

        if train_num[k[s]][s] == round(class_num[s] / ratio):
            k[s] = k[s] + 1 if k[s] + 1 < ratio else k[s]

    print()
    print(class_num)
    print(train_num)
    print(f'There are {len(no_label)} Linages have no label in Label File: {no_label}')
    print(f"There are {len(multi_label)} Linages have multiple labels: {multi_label}")
    print(f'There are {len(samll_label)} Access Date have  have the inputs future but no labels')
    print(f'There are {len(less_label)} Access Date does not have {exp_days} days.')
    print(
        f'There are {len(lag_label)} Access Date that time lag to recoder is larger or equal to {round(exp_days / 2)} days.')

    for i in range(ratio):
        exp_label[i] = np.stack(exp_label[i], axis=0)
        print(len(exp_set[i]), exp_label[i].shape)

    num_data = int(len(exp_set[-1]))

    flat_exp_set = []
    flat_linage = []
    for l in exp_set:
        flat_exp_set += l
    for l in linage:
        flat_linage += l

    return train_set, train_label, linage, flat_exp_set, flat_linage, np.concatenate(exp_label,
                                                                                     axis=0), samll_label, less_label, lag_label, num_data


def generate_cross_validation_one_country_new_by_linage(dataset, label, dict_var, sbar, max_day=365, days=5,
                                                        exp_days=14,
                                                        future=14, ratio=5, position=False,
                                                        features=None, input_feature=3, out_feature=3,
                                                        random_split=True,
                                                        token=-4, prefix_padding=True):
    assert future >= 0

    test_set = defaultdict(list)
    test_label = defaultdict(list)

    exp_set = defaultdict(list)
    exp_label = defaultdict(list)

    date_sort = []

    for key, date in dataset.items():
        min_access = min(date.keys())
        date_sort.append([key, min_access])

    date_sort.sort(key=lambda x: x[-1][-1])

    no_label = []
    multi_label = []
    samll_label = []
    less_label = []
    lag_label = []

    for key_day in date_sort:
        key = key_day[0]
        access_date = dataset[key]

        if key not in label.keys():
            no_label.append(key)
            break

        min_access = min(access_date.keys())

        for access_days, data in access_date.items():

            relevant_day = access_days[1] - min_access[1] + 1

            if relevant_day > max_day:
                break

            label_pango = label[key]
            window_end = access_days[1]
            ind_future = window_end + future

            l = label_pango[label_pango[:, 0] == ind_future]

            if l.shape[0] > 1:
                multi_label.append(key)
            elif l.shape[0] == 0:
                samll_label.append((key, access_days))
                continue

            lag_day = int(access_days[1] - data[-1, 0])
            record_day = int(days - lag_day)

            if lag_day > (days / 2):
                lag_label.append((key, access_days[0]))
                continue

            if data.shape[0] < record_day:
                less_label.append((key, access_days[0]))

                if not prefix_padding:
                    continue

                start_day = int(access_days[1] - days) + 1
                start_record = int(data[0, 0] - start_day)

                pad1 = np.arange(start_day, int(data[0, 0]))[:, np.newaxis]
                pad2 = np.ones([start_record, data.shape[1] - 3]) * token
                pad3 = np.zeros([start_record, 2]) + data[0, -2:]
                pad_f = np.concatenate([pad1, pad2, pad3], axis=1)

                pad1 = np.arange(int(data[-1, 0]), access_days[1])[:, np.newaxis] + 1
                pad2 = np.ones([lag_day, data.shape[1] - 3]) * token
                pad3 = np.zeros([lag_day, 2]) + data[0, -2:]
                pad_b = np.concatenate([pad1, pad2, pad3], axis=1)

                value = np.concatenate([pad_f, data, pad_b], axis=0)
            else:
                pad1 = np.arange(int(data[-1, 0]), access_days[1])[:, np.newaxis] + 1
                pad2 = np.ones([lag_day, data.shape[1] - 3]) * token
                pad3 = np.zeros([lag_day, 2]) + data[-record_day, -2:]
                pad = np.concatenate([pad1, pad2, pad3], axis=1)
                value = np.concatenate([data[-record_day:], pad], axis=0)

            if position:
                p = relevant_day * np.ones([days, 1])
                if features:
                    test_set[key].append(np.concatenate([value[:, features], p], axis=1))
                else:
                    test_set[key].append(np.concatenate([value, p], axis=1))
            else:
                if features:
                    test_set[key].append(value[:, features])
                else:
                    test_set[key].append(value)

            test_label[key].append(l[0, out_feature])

            exp_set[key].append(data[int(np.max([0, data.shape[0] - exp_days])):, input_feature])
            exp_label[key].append(l[0, out_feature])

    print()
    print(f'There are {len(no_label)} Linages have no label in Label File: {no_label}')
    print(f"There are {len(multi_label)} Linages have multiple labels: {multi_label}")
    print(f'There are {len(samll_label)} Access Date have  have the inputs future but no labels')
    print(f'There are {len(less_label)} Access Date does not have {days} days.')
    print(
        f'There are {len(lag_label)} Access Date that time lag to recoder is larger or equal to {round(days / 2)} days.')

    test_set2 = {}
    test_label2 = {}
    for i in test_set.keys():
        test_set2[i] = np.stack(test_set[i], axis=0).reshape([len(test_set[i]), -1])
        test_label2[i] = np.stack(test_label[i], axis=0)
        print(test_set2[i].shape, test_label2[i].shape)

    return test_set2, test_label2, exp_set, exp_label, samll_label, less_label, lag_label


def time_plot(pred_dict_list, true_dict_list, key_list, seed, save_path):
    fig, ax = plt.subplots(7, 3, figsize=(24, 56), dpi=300)

    color_list = ['b', 'g', 'c', 'r', 'y', 'm']

    for ind, linage in enumerate(key_list):
        a, b = ind // 3, ind % 3

        data_list = []

        for i in range(5):
            data = pred_dict_list[i][linage]
            ax[a, b].plot(data[0], data[1], label=i + 1, c=color_list[i + 1], alpha=0.3)
            data_list.append(data)

        mean_data = np.nanmean(np.stack(data_list), axis=0)
        ax[a, b].plot(mean_data[0], mean_data[1], label='mean', c=color_list[0])

        label = true_dict_list[0][linage]
        for i in range(1, 5):
            assert (true_dict_list[i][linage] == label).all()

        ax[a, b].plot(label[0], label[1], c='k', label='label')

        lims = [
            np.min([ax[a, b].get_ylim()]),  # min of both axes
            np.max([ax[a, b].get_ylim()]),  # max of both axes
        ]

        loss = np.mean((mean_data - label) ** 2)
        ax[a, b].set_title(f'{linage}: {loss}', y=1)
        ax[a, b].legend()

    plt.savefig(save_path + f'time_linage_{seed}.png')
    plt.close()


"""## LOAD"""
# future_split = '2023-12-31'
train_val_split = '2022-12-31'
future_split = '2024-02-28'
# train_val_split = '2023-03-31'

def deal_str_date(str_date):
    real_start_date = '2020-01-01'
    start = datetime.datetime(int(real_start_date.split('-')[0]), int(real_start_date.split('-')[1]),
                              int(real_start_date.split('-')[2]))
    days = (datetime.datetime(int(str_date.split('-')[0]), int(str_date.split('-')[1]),
                              int(str_date.split('-')[2])) - start).days
    return days


def first_process_data(df_sort, country, country_token, consecutive_days):
    # Generate a dictionary with 'Pango' as keys and all other columns as values
    result_dict = defaultdict(dict)
    for pango, group in df_sort.groupby('Pango'):
        for access_date, date_group in group.groupby('AccessDate'):
            data = date_group.values.tolist()

            # Find Error data
            if data[-1][4] > data[-1][2]:
                print(country, pango, access_date, data[-1][3])
                continue

            # Loop over the sublists to find the first sublist with consecutive dates
            if consecutive_days > 0:
                prev_date, prev_index = None, None
                consecutive_count = 1

                start_correct_date = None

                for i, sublist in enumerate(data):
                    if prev_date is not None and sublist[4] == data[i - 1][4] + 1:
                        consecutive_count += 1
                        if consecutive_count == consecutive_days:
                            start_correct_date = prev_index
                            break
                    else:
                        consecutive_count = 1
                        prev_date = sublist
                        prev_index = i

                if start_correct_date is not None:
                    data = [x[4:] for x in data[start_correct_date:]]
                else:
                    continue
            else:
                data = [x[4:] for x in data]

            # loop over the sublists to fill any gaps in dates
            prev_date = None
            new_data = []

            for sublist in data:
                date = sublist[0]

                if prev_date is not None and date > prev_date + 1:
                    gap = date - prev_date - 1
                    new_lol = []
                    for i in range(prev_date + 1, date):
                        new_list = [i] + [np.nan] * (len(sublist) - 1)
                        new_lol.append(new_list)
                    new_data += new_lol

                new_data.append(sublist)
                prev_date = date

            new_data = np.array(new_data)
            c = np.repeat(country_token[country][np.newaxis,], new_data.shape[0], axis=0)

            access_day = (access_date, deal_str_date(access_date))
            result_dict[country + '_' + pango][access_day] = np.concatenate([new_data, c], axis=1)

    return result_dict


def load_data_new_time(path, country, suffix='', consecutive_days=2, split_date=train_val_split, new_split=True):
    # '2022-07-31'
    def default_array():
        return np.array([0, 1])
    country_token = defaultdict(default_array)
    country_token['GBR'] = np.array([0, 1])
    country_token['USA'] = np.array([1, 0])
    # country_token = {'GBR': np.array([0, 1]), 'USA': np.array([1, 0]), 'DEU': np.array([0, 0]), 'DNK': np.array([0, 0])}

    file_path = os.path.join(path, 'freqs_bySubDate_' + country + suffix + '.csv')

    df = pd.read_csv(file_path)

    # Find start date
    start_date = df['CollectionDate'].min()
    print(start_date)

    split = deal_str_date(split_date)

    # Sort the data by 'Pango', 'AccessDate', 'CollectionDate'
    df_sort = df.sort_values(['Pango', 'AccessDate', 'CollectionDate'])[
        ['Pango', 'AccessDate', 'CollectionDate', 'totalSeqs', 'VarSeqs', 'VarFreq']]

    # Change date to abosulte value
    df_sort['AccessDay'] = df_sort['AccessDate'].apply(deal_str_date)
    df_sort['CollectionDay'] = df_sort['CollectionDate'].apply(deal_str_date)

    df_sort = df_sort[
        ['Pango', 'AccessDate', 'AccessDay', 'CollectionDate', 'CollectionDay', 'totalSeqs', 'VarSeqs', 'VarFreq']]

    # Identify rows where  has AccessDay >= threshold
    # df_train = df_sort[df_sort['AccessDay'] <= split]
    # df_test = df_sort[df_sort['AccessDay'] > split]

    # Identify Pangos where at least one row has AccessDay >= threshold
    # pangos_to_move = df_sort[df_sort['AccessDay'] >= split]['Pango'].unique()
    # df_train = df_sort[~df_sort['Pango'].isin(pangos_to_move)]
    # df_test = df_sort[df_sort['Pango'].isin(pangos_to_move)]

    # Calculate the proportion of rows for each 'Pango' that have 'AccessDay' greater than or equal to the threshold
    # Identify Pangos where this proportion is greater than or equal to 0.5
    # proportion_series = df_sort[df_sort['AccessDay'] < split].groupby('Pango').size() / df_sort.groupby('Pango').size()
    # pangos_to_move = proportion_series[proportion_series >= 0.5].index.tolist()
    # df_train = df_sort[~df_sort['Pango'].isin(pangos_to_move)]
    # df_test = df_sort[df_sort['Pango'].isin(pangos_to_move)]

    # result_dict_train = first_process_data(df_train, country, country_token, consecutive_days)
    # result_dict_test = first_process_data(df_test, country, country_token, consecutive_days)

    if new_split:
        result_dict = first_process_data(df_sort, country, country_token, consecutive_days)

        result_dict_train, result_dict_test = {}, {}
        for pango_name, days_dict in result_dict.items():
            condition = any(day[1] >= split for day in days_dict.keys())

            if condition:
                result_dict_test[pango_name] = days_dict
            else:
                result_dict_train[pango_name] = days_dict
    else:
        df_train = df_sort[df_sort['AccessDay'] <= split]
        df_test = df_sort[df_sort['AccessDay'] > split]
        result_dict_train = first_process_data(df_train, country, country_token, consecutive_days)
        result_dict_test = first_process_data(df_test, country, country_token, consecutive_days)

    return start_date, result_dict_train, result_dict_test, df_sort


def remove_isolated_points(arr, threshold=4, window=3):
    if len(arr) <= 3:
        return []
    # Create a boolean mask for NaNs
    nan_mask = np.isnan(arr[:, -1])
    # Create a copy of the array for modifications
    arr_mod = arr.copy()
    arr_ind = np.arange(0, len(arr))

    for i in range(0, len(arr)):
        window_mask = nan_mask[max(0, i - window): min(len(arr), i + window + 1)]
        if arr_mod[i, -1] and np.sum(window_mask) / (len(window_mask) - 1) >= threshold / (window * 2):
            arr_mod[i, -1] = np.nan

    # Remove NaNs at the start and end
    while np.isnan(arr_mod[0, -1]):
        arr_mod = arr_mod[1:]
        arr_ind = arr_ind[1:]
    while np.isnan(arr_mod[-1, -1]):
        arr_mod = arr_mod[:-1]
        arr_ind = arr_ind[:-1]

    return arr_ind


def count_local_extrema(array):
    # Calculate the difference between consecutive elements
    diff = np.diff(array)
    # Calculate the sign of the difference
    signs = np.sign(diff)
    # Calculate the difference between consecutive signs
    sign_diff = np.diff(signs)

    # Local maxima occur where the sign changes from positive to negative, so the difference is -2
    local_maxima = np.sum(sign_diff == -2)
    # Local minima occur where the sign changes from negative to positive, so the difference is 2
    local_minima = np.sum(sign_diff == 2)

    return local_maxima + local_minima


def load_label_smooth(path, country, s=30, k=4, output_noise_level=False, remove_isolate=True):
    file_path = os.path.join(path, 'freqs_final_' + country + '.csv')

    df = pd.read_csv(file_path)

    # Find start date
    start_date = df['CollectionDate'].min()
    print(start_date)

    real_start_date = '2020-01-01'
    start = datetime.datetime(int(real_start_date.split('-')[0]), int(real_start_date.split('-')[1]),
                              int(real_start_date.split('-')[2]))

    def deal_str_date(str_date, start=start):
        days = (datetime.datetime(int(str_date.split('-')[0]), int(str_date.split('-')[1]),
                                  int(str_date.split('-')[2])) - start).days
        return days

    # Sort the data by 'Pango', 'AccessDate', 'CollectionDate'
    df_sort = df.sort_values(['Pango', 'CollectionDate'])[
        ['Pango', 'CollectionDate', 'totalSeqs', 'VarSeqs', 'VarFreq']]

    # Change date to abosulte value
    df_sort['CollectionDay'] = df_sort['CollectionDate'].apply(deal_str_date)

    df_sort = df_sort[['Pango', 'CollectionDate', 'CollectionDay', 'totalSeqs', 'VarSeqs', 'VarFreq']]

    # Generate a dictionary with 'Pango' as keys and all other columns as values
    raw_dict = {}
    original_dict = {}
    log_dict = {}
    interpolated_dict = {}
    mask_dict = {}
    smoothed_dict = {}

    noise_level = []
    noise_ind = []
    noise_ind2 = []

    missing_list = set()
    remove_list = set()
    num_less = 0
    for pango, group in df_sort.groupby('Pango'):
        data = group.values.tolist()

        data = [x[2:] for x in data]

        prev_date = None
        new_data = []
        raw = []

        for i, sublist in enumerate(data):
            date = sublist[0]

            if prev_date is not None and date > prev_date + 1:
                missing_list.add(pango)
                gap = date - prev_date - 1
                new_lol = []
                for i in range(prev_date + 1, date):
                    new_list = [i] + [np.nan] * (len(sublist) - 1)
                    new_lol.append(new_list)
                new_data += new_lol

            new_data.append(sublist)
            prev_date = date

            raw.append(sublist)

        original_data = np.array(new_data)
        raw_data = np.array(raw)

        if remove_isolate:
            try:
                pango_ind = remove_isolated_points(original_data, threshold=4, window=3)
            except:
                remove_list.add(pango)
                continue
            if not len(pango_ind):
                remove_list.add(pango)
                continue
            original_data = original_data[pango_ind[0]:pango_ind[-1] + 1]

            while raw_data[0, 0] < original_data[0, 0]:
                raw_data = raw_data[1:]
            while raw_data[-1, 0] > original_data[-1, 0]:
                raw_data = raw_data[:-1]

        pango_data = original_data.copy()
        pango_data[:, 1:] = np.log10(original_data[:, 1:])

        interpolated_pango = pango_data.copy()
        interpolated_mask = np.full(pango_data.shape[0], True)
        smoothed_pango = pango_data.copy()

        less_data = False
        for i in range(1, 4):
            mask = ~np.isnan(pango_data[:, i])

            if np.sum(mask) <= 1:
                less_data = True
                num_less += 1
                break

            interpolated_data = pd.Series(pango_data[:, i]).interpolate(method='linear').to_numpy().squeeze()

            rules = (pango in ['BA.1.1']) and country == 'USA'
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                spoint = 3
                spline = UnivariateSpline(pango_data[:, 0], interpolated_data, s=spoint, k=np.amin(
                    [np.sum(mask) - 1, k]))  # s is the smoothing factor (0.1,4) (0.09,4) (0.085,4) (0.0625,5)
                if not rules:
                    while count_local_extrema(spline(pango_data[:, 0])) > 5 and remove_isolate:
                        spoint += 8
                        spline = UnivariateSpline(pango_data[:, 0], interpolated_data, s=spoint,
                                                  k=np.amin([np.sum(mask) - 1, k]))
                else:
                    while count_local_extrema(spline(pango_data[:, 0])) > 4 and remove_isolate:
                        spoint += 1
                        spline = UnivariateSpline(pango_data[:, 0], interpolated_data, s=spoint,
                                                  k=np.amin([np.sum(mask) - 1, 2]))
            smoothed_y = spline(pango_data[:, 0])  # np.maximum(spline(pango_data[:,0]), 0)

            interpolated_pango[:, i] = interpolated_data
            if i > 1:
                assert (np.logical_and(interpolated_mask, mask) == interpolated_mask).all()
            else:
                interpolated_mask = np.logical_and(interpolated_mask, mask)
            smoothed_pango[:, i] = smoothed_y

            if i == 3:
                noise_level.append(np.abs(pango_data[mask][:, 3] - smoothed_y[mask]))
                noise_ind.append(pango_data[mask][:, 0])
                noise_ind2.append(pango_data[mask][:, 3])

        if less_data:
            continue

        raw_dict[country + '_' + pango] = raw_data
        original_dict[country + '_' + pango] = original_data
        log_dict[country + '_' + pango] = pango_data
        interpolated_dict[country + '_' + pango] = interpolated_pango
        mask_dict[country + '_' + pango] = interpolated_mask
        smoothed_dict[country + '_' + pango] = smoothed_pango

    print(country, f' Label Has {int(len(missing_list))} Missing')
    print(num_less)
    print(country, f' {int(len(remove_list))} Removed')
    print(len(raw_dict))

    noise_level = np.concatenate(noise_level, axis=0)
    noise_ind = np.concatenate(noise_ind, axis=0)
    noise_ind2 = np.concatenate(noise_ind2, axis=0)

    if output_noise_level:
        return start_date, raw_dict, original_dict, log_dict, interpolated_dict, mask_dict, smoothed_dict, df_sort, noise_level, noise_ind, noise_ind2

    return start_date, raw_dict, original_dict, log_dict, interpolated_dict, mask_dict, smoothed_dict, df_sort


def read_concern(path='related_files/concern-GBR-USA.csv', name='Concern'):
    csv.field_size_limit(100000000)
    csvFile = open(path, "r", encoding='utf-8')
    reader = csv.DictReader(csvFile)

    dict_var = {}
    for item in reader:
        dict_var[item['Country'] + '_' + item['Pango']] = float(item[name])
    csvFile.close()

    return dict_var


"""##Process"""


def dist(results_list):
    results = {}

    for l in results_list:
        results[l[0]] = l[1:]

    x1, y1 = [], []
    x2, y2 = [], []

    for v in results.values():
        x1.append(v[0].numpy())
        y1.append(v[2].numpy())

        x2.append(v[1].numpy())
        y2.append(v[3].numpy())

    print('L2:')
    print(
        f'Val: Min: {np.min(x1)}, Max: {np.max(x1)}, Mean: {np.mean(x1)}, Std: {np.std(x1)}, Median: {np.median(x1)}')
    print(
        f'Test: Min: {np.min(y1)}, Max: {np.max(y1)}, Mean: {np.mean(y1)}, Std: {np.std(y1)}, Median: {np.median(y1)}')
    print()

    print('L1:')
    print(
        f'Val: Min: {np.min(x2)}, Max: {np.max(x2)}, Mean: {np.mean(x2)}, Std: {np.std(x2)}, Median: {np.median(x2)}')
    print(
        f'Test: Min: {np.min(y2)}, Max: {np.max(y2)}, Mean: {np.mean(y2)}, Std: {np.std(y2)}, Median: {np.median(y2)}')
    print()


def log_transform(data, k=1, c=0):
    return (np.log10(np.abs(k * data) + c)) * np.sign(data)


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def clean_data(dataset_dict, interp=True, token=-100, nan_indicat=False):
    result_dict = defaultdict(dict)
    indicator = defaultdict(dict)
    drop_list = []

    for pango, pango_value in dataset_dict.items():
        for access_date, value in pango_value.items():
            data_current = value.copy()
            nan_indicator = np.logical_not(np.isnan(data_current))

            # remove nan data above first one
            if interp:
                ind = np.all(np.logical_not(np.isnan(data_current)), axis=-1).argmax()
                data_current = data_current[ind:]
                nan_indicator = nan_indicator[ind:]

            for i in range(1, data_current.shape[-1]):
                y = data_current[:, i]
                nans, x = nan_helper(y)
                if interp:
                    y[nans] = np.interp(x(nans), x(~nans), y[~nans])
                else:
                    y[nans] = token
                data_current[:, i] = y

            result_dict[pango][access_date] = np.array(data_current)
            indicator[pango][access_date] = nan_indicator

    if nan_indicat:
        return result_dict, indicator
    else:
        return result_dict


def clean_label(dataset_dict, interp=True, token=-100):
    result_dict = {}
    drop_list = []

    for pango, value in dataset_dict.items():
        data_current = value.copy()

        # remove nan data above first one
        if interp:
            ind = np.all(np.logical_not(np.isnan(data_current)), axis=-1).argmax()
            data_current = data_current[ind:]

        for i in range(1, data_current.shape[-1]):
            y = data_current[:, i]
            nans, x = nan_helper(y)
            if interp:
                y[nans] = np.interp(x(nans), x(~nans), y[~nans])
            else:
                y[nans] = token
            data_current[:, i] = y

            result_dict[pango] = np.array(data_current)
    return result_dict


def create_padding(start_day, end_day, reference_data, token):
    # Calculate the record length
    record_length = end_day - start_day

    # Create arrays for padding
    pad1 = np.arange(start_day, end_day)[:, np.newaxis]
    pad2 = np.ones([record_length, reference_data.shape[1] - 3]) * token
    pad3 = np.zeros([record_length, 2]) + reference_data[0, -2:]

    # Concatenate to create final padding
    padding = np.concatenate([pad1, pad2, pad3], axis=1)

    return padding


def data_padding(data, days, access_days, key, token, nan_mask=None, prefix_padding=True):
    less_label = []

    lag_day = int(access_days[1] - data[-1, 0])
    record_day = int(days - lag_day)

    if data.shape[0] < record_day:
        less_label.append((key, access_days[0]))

        if not prefix_padding:
            return None, None, less_label

        start_day = int(access_days[1] - days) + 1
        end_day = int(data[0, 0])
        pad_f = create_padding(start_day, end_day, data, token)

        value = np.concatenate([pad_f, data], axis=0)
        nan_mask = np.concatenate([np.full_like(pad_f, 0.), nan_mask], axis=0) if nan_mask is not None else None
    else:
        value = data[-record_day:]
        nan_mask = nan_mask[-record_day:] if nan_mask is not None else None

    end_day = access_days[1] + 1
    start_day = int(data[-1, 0]) + 1
    assert start_day <= end_day
    pad_b = create_padding(start_day, end_day, value, token)
    value = np.concatenate([value, pad_b], axis=0)
    nan_value = np.concatenate([nan_mask, np.full_like(pad_b, 0.)], axis=0) if nan_mask is not None else None

    return value, nan_value, less_label


def generate_cross_validation_one_country_new(dataset, label, dict_var, sbar, max_day=365, days=5, exp_days=14,
                                              future=14, ratio=5, position=False,
                                              features=None, input_feature=3, out_feature=3, random_split=True,
                                              token=-4, prefix_padding=True, smoothed_label=None, noise_level=-1, drop_rule=1, seq_out=False,
                                              nan_ind=None, split_date=future_split):
    assert future >= 0
    split = deal_str_date(split_date)

    train_set = [[] for _ in range(ratio)]
    train_label = [[] for _ in range(ratio)]
    smooth_label = [[] for _ in range(ratio)]
    train_label_smooth = [[] for _ in range(ratio)]
    train_nan = [[] for _ in range(ratio)]

    exp_set, exp_label = [[] for _ in range(ratio)], [[] for _ in range(ratio)]

    linage = [[] for _ in range(ratio)]

    date_sort = []

    for key, date in dataset.items():
        min_access = min(date.keys())
        date_sort.append([key, min_access])

    date_sort.sort(key=lambda x: x[-1][-1])
    if random_split:
        random.shuffle(date_sort)

    class_num = {0: 0, 1: 0, 2: 0, 3: 0}  # 3 is tha pango not has concern score
    all = 0
    for key_day in date_sort:
        key = key_day[0]

        # if key not in label.keys():
        #     break

        if key in dict_var.keys():
            cs = dict_var[key]
            if cs <= sbar[0]:
                s = 0
            elif sbar[0] < cs <= sbar[1]:
                s = 1
            else:
                s = 2
        else:
            s = 3
        class_num[s] += 1
        all += 1

    k = [0, 0, 0, 0]
    train_num = [{0: 0, 1: 0, 2: 0, 3: 0} for _ in range(ratio)]
    no_label = []
    multi_label = []
    samll_label = []
    less_label = []
    lag_label = []
    too_close_label = []
    wrong_label = []
    no_smooth_input = []

    max_label = -1

    for key_day in date_sort:
        key = key_day[0]
        access_date = dataset[key]
        nan_masks = nan_ind[key] if nan_ind else None

        if key not in label.keys():
            no_label.append(key)
            continue

        if key in dict_var.keys():
            cs = dict_var[key]
            if cs <= sbar[0]:
                s = 0
            elif sbar[0] < cs <= sbar[1]:
                s = 1
            else:
                s = 2
        else:
            s = 3

        min_access = min(access_date.keys())

        for access_days, data in access_date.items():

            nan_mask = nan_masks[access_days] if nan_masks else None

            relevant_day = access_days[1] - min_access[1] + 1

            if relevant_day > max_day:
                break

            if access_days[1] < data[-1, 0]:
                wrong_label.append([key, access_days])
                continue

            # value = data[-days:]

            label_pango = label[key]
            if smoothed_label:
                smooth_label_pango = smoothed_label[key]
            window_end = access_days[1]
            ind_future = window_end + future

            if ind_future > split:
                too_close_label.append((key, access_days[0]))
                continue

            l_ind = np.where(label_pango[:, 0] == ind_future)[0]

            if l_ind.shape[0] > 1:
                multi_label.append(key)
            elif l_ind.shape[0] == 0:
                samll_label.append((1, key, access_days))
                continue

            if seq_out:
                l_ind2 = np.where(label_pango[:, 0] == window_end + 1)[0]
                if l_ind2.shape[0] == 0:
                    samll_label.append((key, access_days))
                    continue
                l = label_pango[l_ind2[0]:l_ind[0] + 1]
                if smoothed_label:
                    sl = smooth_label_pango[l_ind2[0]:l_ind[0] + 1]
                if l.shape[0] < future:
                    samll_label.append((key, access_days))
                    continue
            else:
                l = label_pango[l_ind[0]][np.newaxis,]
                if smoothed_label:
                    sl = smooth_label_pango[l_ind[0]][np.newaxis,]
            # l = label_pango[label_pango[:, 0] == ind_future]

            if smoothed_label:
                smooth_ind_end = np.where(smooth_label_pango[:, 0] == window_end)[0]
                smooth_ind_start = np.where(smooth_label_pango[:, 0] == (window_end - days + 1))[0]
                need_pad = (smooth_ind_end.shape[0] == 0) or (smooth_ind_start.shape[0] == 0)

                if window_end < smooth_label_pango[0, 0] or window_end - days + 1 > smooth_label_pango[-1, 0]:
                    no_smooth_input.append((1, key, access_days, smooth_ind_start, smooth_ind_end))
                    continue

                if smooth_ind_end.shape[0] == 0:
                    smooth_ind_end = np.array([smooth_label_pango.shape[0] - 1])
                if smooth_ind_start.shape[0] == 0:
                    smooth_ind_start = np.array([0])

                smooth_l = smooth_label_pango[smooth_ind_start[0]:smooth_ind_end[0] + 1]

                if need_pad:
                    no_smooth_input.append((2, key, access_days, smooth_ind_end, smooth_ind_start))

                    smooth_l, _, _ = data_padding(smooth_l, days, access_days, key, np.nan, None, prefix_padding)

                    if smooth_l is None:
                        continue

            lag_day = int(access_days[1] - data[-1, 0])
            record_day = int(days - lag_day)

            if lag_day >= days or (np.nansum(nan_mask[-max(record_day, 0):, input_feature]) < drop_rule and drop_rule > 0) or (
                    lag_day > (days / 2) and drop_rule < 0):  #
                lag_label.append((key, access_days[0]))
                continue

            value, nan_value, ll = data_padding(data, days, access_days, key, token, nan_mask, prefix_padding)
            less_label += ll

            if value is None:
                continue

            if value.shape[0] != days:
                print(key, access_days)
                print(data[:, 0])

            if position:
                p = relevant_day * np.ones([days, 1])
                if features:
                    train_set[k[s]].append(np.concatenate([value[:, features], p], axis=1))

                else:
                    train_set[k[s]].append(np.concatenate([value, p], axis=1))
            else:
                if features:
                    train_set[k[s]].append(value[:, features])
                else:
                    train_set[k[s]].append(value)

            if nan_ind:
                if position:
                    p = relevant_day * np.ones([days, 1])
                    if features:
                        train_nan[k[s]].append(np.concatenate([nan_value[:, features], p], axis=1))

                    else:
                        train_nan[k[s]].append(np.concatenate([nan_value, p], axis=1))
                else:
                    if features:
                        train_nan[k[s]].append(nan_value[:, features])
                    else:
                        train_nan[k[s]].append(nan_value)

            # if smoothed_label and noise_level >= 0 and (0 >= l[:, out_feature] >= -1 or 1 >= l[:, out_feature] >= 0.1) and smooth_ind_end.size and smooth_ind_start.size:
            #     l2 = smooth_ind_end[0]
            #     l1 = smooth_ind_start[0]
            #     s_value = smooth_label_pango[l1:l2 + 1]
            #     for _ in range(2):
            #         s_value[:, 1:] += np.random.default_rng().normal(0, noise_level,
            #                                                          (s_value.shape[0], s_value.shape[1] - 1))
            #         # value = s_value
            #         value[:, input_feature] = s_value[:, out_feature]
            #         if position:
            #             p = relevant_day * np.ones([days, 1])
            #             if features:
            #                 train_set[k[s]].append(np.concatenate([value[:, features], p], axis=1))
            #             else:
            #                 train_set[k[s]].append(np.concatenate([value, p], axis=1))
            #         else:
            #             if features:
            #                 train_set[k[s]].append(value[:, features])
            #             else:
            #                 train_set[k[s]].append(value)
            #     train_label[k[s]].append(l[:, out_feature])

            linage[k[s]].append((key, s))

            train_label[k[s]].append(l[:, out_feature])
            if smoothed_label:
                smooth_label[k[s]].append(sl[:, out_feature])
                train_label_smooth[k[s]].append(smooth_l[:, out_feature])

            # exp_data = data[int(np.max([0, data.shape[0] - exp_days])):, [0, input_feature]]
            exp_data = value[:, [0, input_feature]]
            exp_set[k[s]].append(exp_data)
            exp_label[k[s]].append(l[:, out_feature])

            if ind_future >= max_label:
                max_label = ind_future

        train_num[k[s]][s] += 1

        if train_num[k[s]][s] == round(class_num[s] / ratio):
            k[s] = k[s] + 1 if k[s] + 1 < ratio else k[s]

    # print()
    print(class_num)
    print(train_num)
    print('Max Label:', max_label)

    # # must remove points
    # print(f'There are {len(no_label)} Linages have no label in Label File:')
    # print(f'There are {len(wrong_label)} Access Dates have wrong labels (access_days[1] < data[-1, 0]): ')
    #
    # # # no use
    # # print(f"There are {len(multi_label)} Linages have multiple labels:")
    # # # not remove points
    # # print(f'There are {len(less_label)} Access Dates does not have {days} days:')
    #
    # # rules can change
    # print(f'There are {len(lag_label)} Access Dates dropped due to rules:')
    #
    # # currently used rules: raw data, split and isolated points
    # print(f'There are {len(samll_label)} Access Dates have the inputs future but no label point (may be since use raw data rather than processed data): ')
    # print(f'There are {len(too_close_label)} Access Dates are later than:')
    # print(f'There are {len(no_smooth_input)} Access Dates remove smooth labels/inputs as isolated points:')

    for i in range(ratio):
        if len(train_set[i]) == 0:
            return [None] * 12

    for i in range(ratio):
        train_set[i] = np.stack(train_set[i], axis=0).reshape([len(train_set[i]), -1]) if train_set[i] else None
        train_label[i] = np.stack(train_label[i], axis=0)
        exp_set[i] = np.stack(exp_set[i], axis=0).reshape([len(exp_set[i]), -1])
        exp_label[i] = np.stack(exp_label[i], axis=0)
        print(train_set[i].shape, train_label[i].shape)
        print(exp_set[i].shape, exp_label[i].shape)
        if smoothed_label:
            smooth_label[i] = np.stack(smooth_label[i], axis=0)
            train_label_smooth[i] = np.stack(train_label_smooth[i], axis=0)
        if nan_ind:
            train_nan[i] = np.stack(train_nan[i], axis=0).reshape([len(train_nan[i]), -1])

    flat_exp_set = []
    flat_linage = []
    # for l in exp_set:
    #     flat_exp_set += l
    for l in linage:
        flat_linage += l

    if smoothed_label and not nan_ind:
        print('1 smoothed_label and not nan_ind')
        return train_set, train_label, train_label_smooth, linage, exp_set, flat_linage, np.concatenate(exp_label,
                                                                                                        axis=0), samll_label, less_label, lag_label
    elif smoothed_label and nan_ind:
        print('2 smoothed_label and nan_ind')
        return train_set, train_label, train_label_smooth, train_nan, smooth_label, linage, exp_set, flat_linage, np.concatenate(exp_label,
                                                                                                                   axis=0), samll_label, less_label, lag_label
    elif not smoothed_label and nan_ind:
        print('3 not smoothed_label and nan_ind')
        return train_set, train_label, train_nan, linage, exp_set, flat_linage, np.concatenate(exp_label, axis=0), samll_label, less_label, lag_label

    print('0 else')
    return train_set, train_label, linage, exp_set, flat_linage, np.concatenate(exp_label,
                                                                                axis=0), samll_label, less_label, lag_label


def generate_cross_validation_one_country_future(dataset, label, dict_var, sbar, max_day=365, days=5, exp_days=14,
                                              future=14, ratio=5, position=False,
                                              features=None, input_feature=3, out_feature=3, random_split=True,
                                              token=-4, prefix_padding=True, smoothed_label=None, noise_level=-1, drop_rule=1, seq_out=False,
                                              nan_ind=None, split_date=future_split):
    assert future >= 0

    split = deal_str_date(split_date)

    train_set = [[] for _ in range(ratio)]
    train_nan = [[] for _ in range(ratio)]

    exp_set = [[] for _ in range(ratio)]

    linage = [[] for _ in range(ratio)]

    date_sort = []

    for key, date in dataset.items():
        min_access = min(date.keys())
        date_sort.append([key, min_access])

    date_sort.sort(key=lambda x: x[-1][-1])
    if random_split:
        random.shuffle(date_sort)

    class_num = {0: 0, 1: 0, 2: 0, 3: 0}  # 3 is tha pango not has concern score
    all = 0
    for key_day in date_sort:
        key = key_day[0]

        # if key not in label.keys():
        #     break

        if key in dict_var.keys():
            cs = dict_var[key]
            if cs <= sbar[0]:
                s = 0
            elif sbar[0] < cs <= sbar[1]:
                s = 1
            else:
                s = 2
        else:
            s = 3
        class_num[s] += 1
        all += 1

    k = [0, 0, 0, 0]
    train_num = [{0: 0, 1: 0, 2: 0, 3: 0} for _ in range(ratio)]
    less_label = []
    lag_label = []
    too_close_label = []
    wrong_label = []

    no_label = []

    max_label = -1

    for key_day in date_sort:
        key = key_day[0]
        access_date = dataset[key]
        nan_masks = nan_ind[key] if nan_ind else None

        if key in dict_var.keys():
            cs = dict_var[key]
            if cs <= sbar[0]:
                s = 0
            elif sbar[0] < cs <= sbar[1]:
                s = 1
            else:
                s = 2
        else:
            s = 3

        if key not in label.keys():
            no_label.append(key)
            continue

        min_access = min(access_date.keys())

        for access_days, data in access_date.items():

            nan_mask = nan_masks[access_days] if nan_masks else None

            relevant_day = access_days[1] - min_access[1] + 1

            if access_days[1] < data[-1, 0]:
                wrong_label.append([key, access_days])
                continue

            window_end = access_days[1]
            ind_future = window_end + future

            label_pango = label[key]
            last_label_day = np.nanmax(label_pango[:, 0])

            if ind_future <= last_label_day:
                too_close_label.append((key, access_days[0]))
                continue

            # if ind_future <= split:
            #     too_close_label.append((key, access_days[0]))
            #     continue

            lag_day = int(access_days[1] - data[-1, 0])
            record_day = int(days - lag_day)

            if lag_day >= days or (np.nansum(nan_mask[-max(record_day, 0):, input_feature]) < drop_rule and drop_rule > 0) or (
                    lag_day > (days / 2) and drop_rule < 0):  #
                lag_label.append((key, access_days[0]))
                continue

            value, nan_value, ll = data_padding(data, days, access_days, key, token, nan_mask, prefix_padding)
            less_label += ll

            if value is None:
                continue

            if value.shape[0] != days:
                print(key, access_days)
                print(data[:, 0])

            if position:
                p = relevant_day * np.ones([days, 1])
                if features:
                    train_set[k[s]].append(np.concatenate([value[:, features], p], axis=1))

                else:
                    train_set[k[s]].append(np.concatenate([value, p], axis=1))
            else:
                if features:
                    train_set[k[s]].append(value[:, features])
                else:
                    train_set[k[s]].append(value)

            if nan_ind:
                if position:
                    p = relevant_day * np.ones([days, 1])
                    if features:
                        train_nan[k[s]].append(np.concatenate([nan_value[:, features], p], axis=1))

                    else:
                        train_nan[k[s]].append(np.concatenate([nan_value, p], axis=1))
                else:
                    if features:
                        train_nan[k[s]].append(nan_value[:, features])
                    else:
                        train_nan[k[s]].append(nan_value)

            linage[k[s]].append((key, s))

            # exp_data = data[int(np.max([0, data.shape[0] - exp_days])):, [0, input_feature]]
            exp_data = value[:, [0, input_feature]]
            exp_set[k[s]].append(exp_data)

            if ind_future >= max_label:
                max_label = ind_future

        train_num[k[s]][s] += 1

        if train_num[k[s]][s] == round(class_num[s] / ratio):
            k[s] = k[s] + 1 if k[s] + 1 < ratio else k[s]

    # print()
    print(class_num)
    print(train_num)
    print('Max Label Future:', max_label)

    # # must remove points
    # print(f'There are {len(no_label)} Linages have no label in Label File:')
    # print(f'There are {len(wrong_label)} Access Dates have wrong labels (access_days[1] < data[-1, 0]): ')
    #
    # # # no use
    # # print(f"There are {len(multi_label)} Linages have multiple labels:")
    # # # not remove points
    # # print(f'There are {len(less_label)} Access Dates does not have {days} days:')
    #
    # # rules can change
    # print(f'There are {len(lag_label)} Access Dates dropped due to rules:')
    #
    # # currently used rules: raw data, split and isolated points
    # print(f'There are {len(samll_label)} Access Dates have the inputs future but no label point (may be since use raw data rather than processed data): ')
    # print(f'There are {len(too_close_label)} Access Dates are later than:')
    # print(f'There are {len(no_smooth_input)} Access Dates remove smooth labels/inputs as isolated points:')

    for i in range(ratio):
        train_set[i] = np.stack(train_set[i], axis=0).reshape([len(train_set[i]), -1])
        exp_set[i] = np.stack(exp_set[i], axis=0).reshape([len(exp_set[i]), -1])
        print(train_set[i].shape)
        print(exp_set[i].shape)
        if nan_ind:
            train_nan[i] = np.stack(train_nan[i], axis=0).reshape([len(train_nan[i]), -1])

    flat_linage = []
    for l in linage:
        flat_linage += l

    if nan_ind:
        return train_set, train_nan, linage, exp_set, flat_linage, less_label, lag_label

    return train_set, linage, exp_set, flat_linage, less_label, lag_label


"""## Plot"""


def time_plot_new(pred_dict_list, true_dict_list, exp_dict_list, original, smoothed, key_list, name_list, color_list, line_format, save_path, name='',
                  test_set=True, country='GBR'):
    # key_list = set()
    # for d in true_dict_list:
    #     key_list.update(d.keys())
    # key_list = list(key_list)

    split_date = train_val_split
    split = deal_str_date(split_date)

    key_list = []
    for key, value in smoothed.items():
        if value.shape[0] == 0 or country not in key:
            continue
        # ind = np.where(value[:, 0] == split)[0]
        # if ind.shape[0] == 0:
        #     if (test_set and value[-1, 0] < split) or (not test_set and value[0, 0] > split):
        #         continue
        # else:
        #     if test_set:
        #         value = value[ind[0]:]
        #     else:
        #         value = value[:ind[0]]
        # if value.shape[0] == 0:
        #     continue
        key_list.append((key, np.nanmax(value[:, -1])))
    key_list.sort(key=lambda x: x[-1], reverse=True)

    valid_len = 0
    for ind, d in enumerate(key_list):
        linage = d[0]
        for j in range(len(true_dict_list)):
            if linage in true_dict_list[j].keys():
                valid_len += 1
                break

    w = 3
    h = min(math.ceil(valid_len / w), 7)  # min(math.ceil(valid_len / w), 27)

    k = 0
    fig, ax = plt.subplots(h, w, figsize=(w * 8, h * 8), dpi=300)
    loss = np.inf

    fig2, ax2 = plt.subplots(1, 1, figsize=(24, 8), dpi=300)

    valid_ind = 0
    for ind, d in enumerate(key_list):
        linage = d[0]
        data_list = []
        has_plot = False

        for i in range(len(true_dict_list)):
            if linage not in true_dict_list[i].keys():
                continue

            has_plot = True

            a, b = valid_ind // w, valid_ind % w
            # print(linage,a,b, end='\n')
            data = pred_dict_list[i][linage]
            ax[a, b].plot(data[0], data[1], line_format[i], label=name_list[i], c=color_list[i], alpha=0.5)
            data_list.append(data)
            l = np.mean(np.abs(data[1] - true_dict_list[i][linage][1]))
            loss = l if l < loss else loss

            if k == 0:
                if valid_ind == 0:
                    ax2.plot(data[0], data[1], line_format[i], label=name_list[i], c=color_list[i], alpha=0.5)
                else:
                    ax2.plot(data[0], data[1], line_format[i], c=color_list[i], alpha=0.5)

        if has_plot:
            # mean_data = np.nanmean(np.stack(data_list), axis=0)
            # ax[a, b].plot(mean_data[0], mean_data[1], label='mean', c=color_list[0])

            if linage in exp_dict_list[0].keys():
                exp = exp_dict_list[0][linage]
                ax[a, b].plot(exp[0], exp[1], '-o', label='exp', c='gray')

                if k == 0:
                    if valid_ind == 0:
                        ax2.plot(exp[0], exp[1], '-o', label='exp', c='gray')
                    else:
                        ax2.plot(exp[0], exp[1], '-o', c='gray')

            raw = original[linage]
            # sorted_indices = np.argsort(label[:, 0])
            # label = label[sorted_indices]
            ax[a, b].plot(raw[:, 0], np.log10(raw[:, -1]), 'o', c='k', label='Raw Label')
            ax[a, b].plot(raw[:, 0], np.log10(raw[:, -1]), '--', c='grey')

            smooth = smoothed[linage]
            # sorted_indices = np.argsort(label[:, 0])
            # label = label[sorted_indices]
            ax[a, b].plot(smooth[:, 0], smooth[:, -1], c='peru', label='Smoothed Label')

            if k == 0:
                if valid_ind == 0:
                    ax2.plot(raw[:, 0], np.log10(raw[:, -1]), '--', c='k', label='Raw Label')
                    # ax2.plot(raw[:, 0], np.log10(raw[:, -1]), '--', c='grey')
                    ax2.plot(smooth[:, 0], smooth[:, -1], c='peru', label='Smoothed Label')
                else:
                    ax2.plot(raw[:, 0], np.log10(raw[:, -1]), '--', c='k')
                    # ax2.plot(raw[:, 0], np.log10(raw[:, -1]), '--', c='grey')
                    ax2.plot(smooth[:, 0], smooth[:, -1], c='peru')

            if np.max(np.log10(raw[:, -1])) >= -1:
                title_color = 'r'
            else:
                title_color = 'k'
            lims = [
                np.min([ax[a, b].get_ylim()]),  # min of both axes
                np.max([ax[a, b].get_ylim()]),  # max of both axes
            ]

            # if test_set:
            #     ax[a, b].set_xlim(left=split)
            # else:
            #     ax[a, b].set_xlim(right=split)
            ax[a, b].set_ylim([-4, 0])
            ax[a, b].set_title(f'{linage}: {loss}', y=1, fontdict={'color': title_color})
            ax[a, b].legend()

            valid_ind += 1

        if has_plot and a == 6 and b == 2:
            k += 1
            fig.savefig(save_path + name + f'time_linage{k}.png')
            plt.close(fig)
            fig, ax = plt.subplots(h, w, figsize=(w * 8, h * 8), dpi=300)
            loss = np.inf
            valid_ind = 0

    fig.savefig(save_path + name + f'time_linage{k}.png')
    plt.close(fig)

    ax2.set_ylim([-4, 0])
    ax2.legend()
    fig2.savefig(save_path + name + f'long_time_linages.png')
    plt.close()


def error_plot_new(pred_dict_list, true_dict_list, original, smoothed, snr, datapoint, input_freq, error, key_list, name_list, color_list, save_path,
                   name='', country='GBR'):
    w = 3
    h = int(len(pred_dict_list))
    fig, ax = plt.subplots(h, w, figsize=(w * 8, h * 8), dpi=300)

    for i in range(len(pred_dict_list)):
        for linage in true_dict_list[i].keys():
            if country not in linage:
                continue
            snr_linage = snr[i][linage]
            datapoint_linage = datapoint[i][linage]
            inputfreq_linage = input_freq[i][linage]
            error_linage = error[i][linage]

            # print(snr_linage.shape, datapoint_linage.shape, error_linage.shape)
            # print([type(el) for el in error_linage[1][:5]])
            # print([type(el) for el in snr_linage[1][:5]])
            # print(error_linage)
            # print(snr_linage)
            ax[i, 0].scatter(snr_linage[1], error_linage[1], label=name_list[i], c=color_list[i], alpha=0.5)
            ax[i, 1].scatter(datapoint_linage[1], error_linage[1], label=name_list[i], c=color_list[i], alpha=0.5)
            ax[i, 2].scatter(inputfreq_linage[1], error_linage[1], label=name_list[i], c=color_list[i], alpha=0.5)

            # if linage in key_list:
            #     print(
            #         f"Linage: {linage}, model: {name_list[i]}, error: {np.mean(error_linage[1])}, \n\t max_snr: {np.max(snr_linage[1])}, mean_snr: {np.mean(snr_linage[1])}, \n\t min_datapoint: {np.min(datapoint_linage[1])}, mean_datapoint: {np.mean(datapoint_linage[1])}")

            ax[i, 0].set_title(f'{name_list[i]}', y=1, fontdict={'color': 'k'})
            ax[i, 1].set_title(f'{name_list[i]}', y=1, fontdict={'color': 'k'})
            ax[i, 2].set_title(f'{name_list[i]}', y=1, fontdict={'color': 'k'})

    plt.savefig(save_path + name + f'error_linage.png')
    plt.close()

    # fig, ax = plt.subplots(h, w, figsize=(w * 8, h * 8), dpi=300)
    #
    # for i in range(len(pred_dict_list)):
    #     for linage in true_dict_list[i].keys():
    # if country not in linage:
    #     continue
    #         snr_linage = snr[i][linage]
    #         datapoint_linage = datapoint[i][linage]
    #         inputfreq_linage = input_freq[i][linage]
    #         error_linage = error[i][linage]
    #
    #         # print(snr_linage.shape, datapoint_linage.shape, error_linage.shape)
    #         # print([type(el) for el in error_linage[1][:5]])
    #         # print([type(el) for el in snr_linage[1][:5]])
    #         # print(error_linage)
    #         # print(snr_linage)
    #         sns.boxplot(x=snr_linage[1], y=error_linage[1], ax=ax[i, 0])
    #         sns.boxplot(x=datapoint_linage[1], y=error_linage[1], ax=ax[i, 1])
    #         sns.boxplot(x=inputfreq_linage[1], y=error_linage[1], ax=ax[i, 2])
    #
    #         ax[i, 0].set_title(f'{name_list[i]}', y=1, fontdict={'color': 'k'})
    #         ax[i, 1].set_title(f'{name_list[i]}', y=1, fontdict={'color': 'k'})
    #         ax[i, 2].set_title(f'{name_list[i]}', y=1, fontdict={'color': 'k'})
    #
    # plt.savefig(save_path + name + f'error_linage_boxplot.png')
    # plt.close()

    s_bin_edges = np.arange(-15, 45, 10)
    s_bin_centers = s_bin_edges[:-1] + 5
    n_bin_edges = np.arange(0, 49, 7)
    n_bin_centers = n_bin_edges[:-1] + 3.5
    f_bin_edges = np.arange(-4.75, 0.25, 0.5)
    f_bin_centers = f_bin_edges[:-1] + 0.25

    fig, ax = plt.subplots(h, w, figsize=(w * 8, h * 8), dpi=300)

    for i in range(len(pred_dict_list)):
        all_snr, all_dp, all_if, all_error = [], [], [], []
        for linage in true_dict_list[i].keys():
            if country not in linage:
                continue
            snr_linage = snr[i][linage]
            datapoint_linage = datapoint[i][linage]
            inputfreq_linage = input_freq[i][linage]
            error_linage = error[i][linage]

            all_snr.append(snr_linage[1])
            all_dp.append(datapoint_linage[1])
            all_if.append(inputfreq_linage[1])
            all_error.append(error_linage[1])
        all_snr = np.concatenate(all_snr)
        all_dp = np.concatenate(all_dp)
        all_if = np.concatenate(all_if)
        all_error = np.concatenate(all_error)

        errors = []
        variances = []
        for j in range(len(s_bin_centers)):
            bin_mask = (all_snr >= s_bin_edges[j]) & (all_snr < s_bin_edges[j + 1])

            bin_errors = all_error[bin_mask]
            bin_mean_error = np.mean(bin_errors) if bin_errors.size else np.nan
            bin_variance = np.var(bin_errors) if bin_errors.size else np.nan
            errors.append(bin_mean_error)
            variances.append(bin_variance)
        errors = np.array(errors)
        variances = np.array(variances)

        ax[i, 0].plot(s_bin_centers, errors, label=name_list[i], c=color_list[i])
        ax[i, 0].errorbar(s_bin_centers, errors, yerr=variances, label=name_list[i], c=color_list[i], alpha=0.2)

        errors = []
        variances = []
        for j in range(len(n_bin_centers)):
            bin_mask = (all_dp >= n_bin_edges[j]) & (all_dp < n_bin_edges[j + 1])

            bin_errors = all_error[bin_mask]
            bin_mean_error = np.mean(bin_errors) if bin_errors.size else np.nan
            bin_variance = np.var(bin_errors) if bin_errors.size else np.nan
            errors.append(bin_mean_error)
            variances.append(bin_variance)
        errors = np.array(errors)
        variances = np.array(variances)

        ax[i, 1].plot(n_bin_centers, errors, label=name_list[i], c=color_list[i])
        ax[i, 1].errorbar(n_bin_centers, errors, yerr=variances, label=name_list[i], c=color_list[i], alpha=0.2)

        errors = []
        variances = []
        for j in range(len(f_bin_centers)):
            bin_mask = (all_if >= f_bin_edges[j]) & (all_if < f_bin_edges[j + 1])

            bin_errors = all_error[bin_mask]
            bin_mean_error = np.mean(bin_errors) if bin_errors.size else np.nan
            bin_variance = np.var(bin_errors) if bin_errors.size else np.nan
            errors.append(bin_mean_error)
            variances.append(bin_variance)
        errors = np.array(errors)
        variances = np.array(variances)

        ax[i, 2].plot(f_bin_centers, errors, label=name_list[i], c=color_list[i])
        ax[i, 2].errorbar(f_bin_centers, errors, yerr=variances, label=name_list[i], c=color_list[i], alpha=0.2)

        ax[i, 0].set_title(f'{name_list[i]}', y=1, fontdict={'color': 'k'})
        ax[i, 1].set_title(f'{name_list[i]}', y=1, fontdict={'color': 'k'})
        ax[i, 2].set_title(f'{name_list[i]}', y=1, fontdict={'color': 'k'})

    plt.savefig(save_path + name + f'error_linage_errorbar.png')
    plt.close()

    print('Regress')
    for i in range(len(pred_dict_list)):
        all_snr, all_dp, all_if, all_error = [], [], [], []
        for linage in true_dict_list[i].keys():
            snr_linage = snr[i][linage]
            datapoint_linage = datapoint[i][linage]
            inputfreq_linage = input_freq[i][linage]
            error_linage = error[i][linage]

            all_snr.append(snr_linage[1])
            all_dp.append(datapoint_linage[1])
            all_if.append(inputfreq_linage[1])
            all_error.append(error_linage[1])
        all_snr = np.concatenate(all_snr)
        all_dp = np.concatenate(all_dp)
        all_if = np.concatenate(all_if)
        all_error = np.concatenate(all_error)

        try:
            print(name_list[i])

            X = sm.add_constant(np.column_stack((all_snr, all_dp, all_if)))

            # Fit the model
            model = sm.OLS(all_error, X).fit()

            # Display the model summary
            print(model.summary())

            # Accessing the coefficients:
            coef_x1, coef_x2, coef_x3 = model.params[1:4]
            print(f"coef_x1: {coef_x1}, coef_x2: {coef_x2}, coef_x3: {coef_x3}")
            print()
        except:
            print('Cannot Regress')
            break


def plot_cumulative_stacked_filled_area(datasets, x_values, colors, ax, title):
    # Sorting datasets based on their starting x position
    dataset_names = list(datasets.keys())
    dataset_names.sort(key=lambda name: x_values[name][0])

    # Create a combined list of all unique x values
    all_x = sorted(set(np.concatenate(list(x_values.values()))))

    # Base fill level for each x position (initialized to zero)
    base_fill = np.zeros_like(all_x)

    # Interpolate and update datasets for all x values, and then compute the cumulative values
    interpolated_datasets = {}
    for idx, name in enumerate(dataset_names):
        color = colors[name]
        x = x_values[name]
        y = datasets[name]

        # Interpolate y values for the current dataset for all x positions
        y_interp = np.interp(all_x, x, y, left=0, right=0)

        # Update y-values to be cumulative sum of its own and all values of datasets below it
        y_cumulative = base_fill + y_interp

        #  Fill between the cumulative y-values of the current dataset and the base
        ax.fill_between(all_x, base_fill, y_cumulative, color=color, alpha=0.3, label=name)

        # Store the interpolated dataset
        interpolated_datasets[name] = y_cumulative

        # Update the base fill level
        base_fill = y_cumulative

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_ylim(0,1)
    ax.grid(True)


def process_dataset(sample_datasets, sample_x_values):
    def softmax(x):
        """Compute the softmax of vector x."""
        exp_x = np.power(10, x)  # for numerical stability
        return exp_x / exp_x.sum(axis=0, keepdims=True)

    result = copy.deepcopy(sample_datasets)

    all_y_sample = {}
    for name, y in sample_datasets.items():
        for xi, yi in zip(sample_x_values[name], y):
            if xi not in all_y_sample:
                all_y_sample[xi] = []
            all_y_sample[xi].append(yi)

    # Compute softmax for each x position
    for xi, ys in all_y_sample.items():
        all_y_sample[xi] = softmax(np.array(ys))

    # Convert numpy arrays to lists for easy manipulation
    for xi in all_y_sample.keys():
        all_y_sample[xi] = list(all_y_sample[xi])

    # Update the sample datasets with softmaxed values
    for name in sample_datasets.keys():
        for i, xi in enumerate(sample_x_values[name]):
            if xi in all_y_sample and len(all_y_sample[xi]) > 0:
                result[name][i] = all_y_sample[xi].pop(0)

    return result


def trans_dict(original, lineage_dict, ground=True, log=True):
    transed = {}
    transed_x = {}
    colors_dict = {}

    for k, v in lineage_dict.items():
        colors_dict[k] = v[1]

        y = []
        x = []

        for l in v[0]:
            if l in original.keys():
                data = original[l]
                if ground:
                    if log:
                        y.append(np.log10(data[:, -1]))
                    else:
                        y.append(data[:, -1])
                    x.append(data[:, 0])
                else:
                    y.append(data[1])
                    x.append(data[0])

        if x and y:
            result_dict = {}
            for idx, vals in zip(x, y):
                for i, v in zip(idx, vals):
                    if i in result_dict:
                        result_dict[i] = np.log10(np.power(10, result_dict[i]) + np.power(10, v))
                    else:
                        result_dict[i] = v

            # Convert dictionary back to arrays if needed
            index_array = np.array(list(result_dict.keys()))
            value_array = np.array(list(result_dict.values()))

            sorted_indices = np.argsort(index_array)
            index_array = index_array[sorted_indices]
            value_array = value_array[sorted_indices]

            transed[k] = value_array
            transed_x[k] = index_array

    return transed, transed_x, colors_dict


def time_plot_area(pred_dict_list, true_dict_list, exp_dict_list, original, smoothed, name_list, save_path, name='', country='GBR'):
    split_date = train_val_split # '2022-12-31'
    split = deal_str_date(split_date)

    from clusters_dict import clusters

    lineage_dict = {}
    used_keys = ['pango_lineages', 'display_name', 'col']
    defined_keys = []
    for k, v in clusters.items():
        if not set(used_keys).issubset(set(v.keys())):
            continue
        pango_lineages = v['pango_lineages']
        lineage = []
        for l in pango_lineages:
            matching_keys = find_keys_with_special_prefix(true_dict_list[0], country + '_' + l["name"])
            if not matching_keys:
                continue
            for m in matching_keys:
              lineage.append(m)
              defined_keys.append(m)

        display_name = v['display_name']
        col = v['col']

        lineage_dict[display_name] = [lineage, col]

    all_key = []
    for linage in true_dict_list[0].keys():
      if country in linage:
          all_key.append(linage)
    other_key = list(set(all_key) - set(defined_keys))
    new_color = ['#d0d5b9', '#b42c8e', '#09fdff', '#cf305f', '#b1d345', '#247c26', '#f60796', '#f0164d', '#c0ac37', '#2a1e77', '#e4590a', '#5f0cac', '#3ad03a']
    lineage_dict['Other'] = [other_key, new_color[-1]]

    w = 1
    h = math.ceil( (len(name_list) + 3) / w)  # min(math.ceil(valid_len / w), 27)
    fig, ax = plt.subplots(h, w, figsize=(w * 12, h * 4), dpi=300) # , dpi=300

    k = 0
    valid_ind = 0

    transed, transed_x, colors_dict = trans_dict(original, lineage_dict, ground=True, log=True)
    transed = process_dataset(transed, transed_x)
    a, b = valid_ind // w, valid_ind % w
    plot_cumulative_stacked_filled_area(transed, transed_x, colors_dict, ax[a], 'Raw')
    valid_ind += 1

    transed, transed_x, colors_dict = trans_dict(smoothed, lineage_dict, ground=True, log=False)
    transed = process_dataset(transed, transed_x)
    a, b = valid_ind // w, valid_ind % w
    plot_cumulative_stacked_filled_area(transed, transed_x, colors_dict, ax[a], 'Smoothed')
    valid_ind += 1

    transed, transed_x, colors_dict = trans_dict(exp_dict_list[0], lineage_dict, ground=False, log=False)
    transed = process_dataset(transed, transed_x)
    a, b = valid_ind // w, valid_ind % w
    plot_cumulative_stacked_filled_area(transed, transed_x, colors_dict, ax[a], 'Exp')
    valid_ind += 1

    for i in range(len(pred_dict_list)):
        transed, transed_x, colors_dict = trans_dict(pred_dict_list[i], lineage_dict, ground=False, log=False)
        transed = process_dataset(transed, transed_x)
        a, b = valid_ind // w, valid_ind % w
        plot_cumulative_stacked_filled_area(transed, transed_x, colors_dict, ax[a], name_list[i])
        valid_ind += 1

    # Collect handles and labels from all subplots
    # handles, labels = [], []
    # for axes in ax.ravel():
    #     ha, l = axes.get_legend_handles_labels()
    #     handles.extend(ha)
    #     labels.extend(l)
    # handles = list(set(handles))
    # labels = list(set(labels))
    handles, labels = ax[3].get_legend_handles_labels()

    xlims = ax[3].get_xlim()
    # Set the xlim for all subplots based on ax[1,0]'s xlim
    for i in range(h):
        for j in range(w):
            ax[i].set_xlim(xlims)

    # Create centralized legend
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.9, 0.9))
    plt.savefig(save_path + name + 'area.png')


def save_all(pred_dict_list, true_dict_list, true_smooth_dict_list, exp_dict_list, raw, smoothed, snr_dict_list, datapoint_dict_list, inputfreq_dict_list, error_dict_list,
             large_pango, name_list, colors, line_format, save_path, phase, future_list, pred_future_dict_list=None, datapoint_future_dict_list=None, inputfreq_future_dict_list=None, exp_dict_future_list=None, file_name=''):
    args_dict = locals().copy()

    # Remove the key for save_path since we don't want to store it twice
    del args_dict['save_path']
    del args_dict['file_name']
    print(save_path + file_name + '.pkl.gz')

    # Save the dictionary using gzip and pickle
    with gzip.open(save_path + file_name + '.pkl.gz', "wb") as f: #all_results_best_14_21_28_35_42_60_all_country
            pickle.dump(args_dict, f)


def save_noise_level(noise_true_dict_list, noise_true_smooth_dict_list, save_path, file_name=''):
    args_dict = locals().copy()

    # Remove the key for save_path since we don't want to store it twice
    del args_dict['save_path']
    del args_dict['file_name']
    print(save_path + file_name + '.pkl.gz')

    # Save the dictionary using gzip and pickle
    with gzip.open(save_path + file_name + '_noise.pkl.gz', "wb") as f:  # all_results_best_14_21_28_35_42_60_all_country
        pickle.dump(args_dict, f)

def load_all(filename):
    with gzip.open(filename, "rb") as f:
        data = pickle.load(f)
    return data
