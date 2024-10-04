# -*- coding: utf-8 -*-

import argparse
import copy
import os
import sys
import math
import bisect
import random
import numpy as np
from datetime import datetime
from collections import defaultdict
import datetime
import csv
import matplotlib.pyplot as plt

from torch.utils.data.dataloader import default_collate
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LinearRegression

import network
from utilities import *
from network import *
import errno
from scipy.stats import gaussian_kde

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--num_epochs', default=500, type=int)
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float, help='learning_rate')
    parser.add_argument('-wd', '--weight_decay', default=0.05, type=float, help='learning_rate')
    parser.add_argument('--dropout', default=0., type=float, help='learning_rate')

    parser.add_argument('--norm', default=True, type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument('--lr_scheduler', default=True, type=bool, action=argparse.BooleanOptionalAction)

    parser.add_argument('--day', default=42, type=int)
    parser.add_argument('--skip_days', default=14, type=int)
    parser.add_argument('--future', default=35, type=int)
    parser.add_argument('--max_day2', default=3650, type=int)

    parser.add_argument('--use_feature', default='all2', type=str)
    parser.add_argument('--used_model', default='transformer_encoder2', type=str)
    parser.add_argument('--loss', default='combain', type=str)
    parser.add_argument('--para_noise_level', default=1e-4, type=float)
    parser.add_argument('--label_type', default='smooth_raw', type=str)

    parser.add_argument('--dims', nargs='+', default=[8], type=int)
    parser.add_argument('--cv', default=True, type=bool, action=argparse.BooleanOptionalAction)

    parser.add_argument('--seed_list', nargs='+', default=[], type=int)
    parser.add_argument('--save_model', default=True, type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument('--ckpt_name', default='', type=str)
    parser.add_argument('--test_on_new', default=True, type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument('--data_phase', default='test', type=str)
    parser.add_argument('--drop_rule', default=2, type=int)
    parser.add_argument('--time_mask', default=True, type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument('--new_split', default=True, type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument('--dataset_version', default='v2', type=str)

    parser.add_argument('--prefix_padding', default=True, type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument('--interp', default=True, type=bool, action=argparse.BooleanOptionalAction)

    parser.add_argument('--save_all', default=True, type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument('--test_future', default=False, type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument('--train_process', default=True, type=bool, action=argparse.BooleanOptionalAction)

    parser.add_argument('--k1', default=-1, type=float)
    parser.add_argument('--k2', default=-2, type=float)
    parser.add_argument('--no_weight_ensamble', default=True, type=bool, action=argparse.BooleanOptionalAction)

    parser.add_argument('--ckpt_dir', default='./ckpt',
                        help='path where to save and load ckpt')
    parser.add_argument('--anno-path', default='./dataset',
                        help='path where to load data')
    parser.add_argument('--output_dir', default='./result',
                        help='path where to save result')
    parser.add_argument('--save_file_name', default='all_results_best_14_21_28_35_42_60_latest3_noise',
                        help='name of results file')

    return parser


def compute_snr_and_valid_count_np(a, b, c):
    batch_size = a.shape[0]
    snr = np.zeros(batch_size)
    valid_count = np.zeros(batch_size, dtype=int)
    input_freq = np.zeros(batch_size)

    for i in range(batch_size):
        # Compute noise as the difference between noisy and true signal
        noise = a[i] - b[i]

        # Only consider valid data points
        valid_a = a[i][c[i]]
        valid_b = b[i][c[i]]
        valid_noise = noise[c[i]]

        # Calculate power of signal and noise
        power_signal = np.nanmean(valid_b ** 2)
        power_noise = np.nanmean(valid_noise ** 2)

        # Avoid divide by zero error
        power_noise = np.where(power_noise == 0, np.ones_like(power_noise) * 1e-8, power_noise)

        # Compute SNR in dB
        snr[i] = 10 * np.log10(power_signal / power_noise)

        # Compute number of valid points where c is True
        valid_count[i] = np.nansum(c[i])

        input_freq[i] = np.nanmean(valid_a)

    assert not np.isnan(snr).any(), f"SNR contains NaN values. {snr}"
    assert not np.isnan(valid_count).any(), f"Valid Count contains NaN values. {valid_count}"

    return snr, valid_count, input_freq


def compute_snr_and_valid_count_np_nolabel(a, c):
    batch_size = a.shape[0]
    valid_count = np.zeros(batch_size, dtype=int)
    input_freq = np.zeros(batch_size)

    for i in range(batch_size):
        # Compute noise as the difference between noisy and true signal

        # Only consider valid data points
        valid_a = a[i][c[i]]

        # Compute number of valid points where c is True
        valid_count[i] = np.nansum(c[i])

        input_freq[i] = np.nanmean(valid_a)

    assert not np.isnan(valid_count).any(), f"Valid Count contains NaN values. {valid_count}"

    return valid_count, input_freq


def test(args, x_test, y_test, y_test_smooth, smooth_label, nan_ind, test_name, seed, pop_list, save_path_list, ensemble=False, prints=False,
         out_type='time'):
    batch_size = args.batch_size

    x_test = torch.tensor(x_test).float()
    y_test = torch.tensor(y_test).float()
    y_test_smooth = torch.tensor(y_test_smooth).float()
    smooth_label = torch.tensor(smooth_label).float()
    nan_ind = torch.tensor(nan_ind).bool()

    if args.label_type in ['original', 'original_raw', 'log_originallabel', 'smooth_originallabel', 'log_raw', 'raw', 'smooth_raw']:
        y_test = torch.log10(y_test)

    if args.use_feature == 'all':
        day_ind = 1
        if args.used_model == 'transformer_token_encoder':
            day_ind = 5
    elif args.use_feature == 'all2':
        day_ind = 3
    else:
        print('The feature option has not days index')
        sys.exit()

    model_list = []
    for save_path in save_path_list:
        for p, ind in enumerate(pop_list):
            if 'reg_classify' in save_path:
                model = network.__dict__['transformer_reg_classify_encoder'](x_test.shape[-1], 1, args.dims, day=args.day,
                                                                             activation='gelu',
                                                                             norm=args.norm, dropout=args.dropout).to(
                    device)
            elif 'large' in save_path:
                model = network.__dict__['transformer_encoder_larger'](x_test.shape[-1], 1, [36], day=args.day, activation='gelu',
                                                                       norm=args.norm, dropout=args.dropout).to(device)
            elif 'combain' in save_path:
                model = network.__dict__[args.used_model](x_test.shape[-1], 1, args.dims, day=args.day, activation='gelu',
                                                          norm=args.norm, dropout=args.dropout, second=True).to(device)
            elif 'moe' in save_path:
                model = network.__dict__['moe'](x_test.shape[-1], 1, args.dims, day=args.day, activation='gelu',
                                                norm=args.norm, dropout=args.dropout, p=len(pop_list)).to(device)
            elif 'seq' in save_path:
                model = network.__dict__['transformer_seq_encoder'](x_test.shape[-1], 1, args.dims * 2, day=args.day, activation='gelu',
                                                                    norm=args.norm, dropout=args.dropout, p=len(pop_list)).to(device)
            else:
                model = network.__dict__[args.used_model](x_test.shape[-1], 1, args.dims, day=args.day, activation='gelu',
                                                          norm=args.norm, dropout=args.dropout).to(device)
            checkpoint = torch.load(save_path + f'_{p}.pth', map_location='cpu')
            print(save_path + f'_{p}.pth')
            model.load_state_dict(checkpoint['model'])
            model.eval()

            model_list.append(model)

    if ensemble:
        reg_result = ensamble_expReg(x_test, args.day, args.future)
        x_test = torch.cat([x_test, reg_result], dim=-1).float()

    dataset_valid = InverNetDataset_name2(x_test, y_test, y_test_smooth, smooth_label, nan_ind, test_name)

    valid_sampler = SequentialSampler(dataset_valid)

    test_loader = DataLoader(
        dataset_valid, batch_size=batch_size,
        sampler=valid_sampler, num_workers=12,
        pin_memory=True, collate_fn=default_collate)

    # Test the model
    linage_result = defaultdict(list)
    linage_label = defaultdict(list)
    linage_label_smooth = defaultdict(list)
    linage_SNR = defaultdict(list)
    linage_numdata = defaultdict(list)
    linage_inputfreq = defaultdict(list)
    linage_error = defaultdict(list)
    linage_exp = defaultdict(list)
    outlayers = []

    with torch.no_grad():
        pred = []
        classes = []
        true = []
        exp_p = []
        index = []
        index_pred = []
        index_input = []
        time_index = []
        for images, labels, labels_smooth, s_labels, nan, names in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            labels_smooth = labels_smooth.to(device)
            nan = nan.to(device)
            s_labels = s_labels.to(device)

            if 'combain' not in save_path_list[0] and 'seq' not in save_path_list[0]:
                all_outputs = [model(images) for model in model_list]
                num_model = len(save_path_list)
            else:
                recent_outputs = [model(images) for model in model_list[len(pop_list):]]
                recent = [torch.clamp(out[:, 0].unsqueeze(1), max=0) for out in recent_outputs]
                recent = torch.stack(recent, dim=0)
                recent = torch.mean(recent, dim=0)  # batch, 1

                all_outputs = [model(images, recent) for model in model_list[:len(pop_list)]]
                num_model = 1

            outputs = [torch.clamp(out[:, 0].unsqueeze(1), max=0) for out in all_outputs]
            outputs = torch.stack(outputs, dim=0).reshape(num_model, len(pop_list), images.shape[0], 1)
            outputs = torch.mean(outputs, dim=1)  # number of model, batch, 1

            d = images.reshape([images.shape[0], args.day, -1]).detach().cpu().numpy()
            m = nan.reshape([images.shape[0], args.day, -1]).detach().cpu().numpy()

            b = s_labels.reshape([images.shape[0], args.day, -1]).detach().cpu().numpy()[:, :, 0]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                snr, valid_count, input_freq = compute_snr_and_valid_count_np(np.log10(d[:, :, 0]), b, m[:, :, 0])

            exp_pred = []

            for q in range(d.shape[0]):
                m_nan = m[q, :, 0]
                x = d[q, :, day_ind]
                v = d[q, :, 0]

                # v = np.log10(v[m_nan])
                v = np.clip(v[m_nan], 1e-6, 1 - 1e-6)
                v = np.log(v / (1 - v))
                x = x[m_nan]

                # m_nan = np.isnan(v)
                # v = v[~m_nan]
                # x = x[~m_nan]

                if v.shape[0] < 2:
                    exp_pred.append(np.nan)
                else:
                    clf = LinearRegression()
                    clf.fit(x.reshape(-1, 1), v.reshape(-1, 1))

                    epred = clf.predict(d[q, -1, day_ind].reshape(1, 1) + args.future)
                    # epred = np.clip(epred, -4, 0)

                    epred = np.clip(epred, -15, 15)
                    epred = np.log10(1 / (1 + np.exp(-epred)))

                    exp_pred.append(epred[0, 0])

            exp_pred = np.stack(exp_pred)

            outputs = torch.mean(outputs, dim=0)

            date_index = images.reshape([images.shape[0], args.day, -1]).detach().cpu().numpy()[:, -1, day_ind] + args.future

            pred.append(outputs.detach().cpu().squeeze(-1))
            true.append(labels.detach().cpu().squeeze(-1))
            exp_p.append(exp_pred)
            time_index.append(date_index)

            index.append(date_index)

            for i, name in enumerate(names):
                linage_result[name].append([date_index[i], outputs[i].detach().cpu().squeeze().numpy()])
                linage_label[name].append([date_index[i], labels[i].detach().cpu().squeeze().numpy()])
                linage_label_smooth[name].append([date_index[i], labels_smooth[i].detach().cpu().squeeze().numpy()])

                linage_SNR[name].append([date_index[i] - args.future, snr[i]])
                linage_numdata[name].append([date_index[i] - args.future, valid_count[i]])
                linage_inputfreq[name].append([date_index[i] - args.future, input_freq[i]])

                linage_error[name].append([date_index[i] - args.future, torch.abs(outputs[i] - labels[i]).detach().cpu().squeeze().numpy()])

                if not np.isnan(exp_pred[i]):
                    linage_exp[name].append([date_index[i], exp_pred[i]])

                if prints and (outputs[i] - labels[i]) ** 2 >= 0.25:
                    outlayers.append(name)

    if prints and len(outlayers):
        print(set(outlayers))

    pred = torch.cat(pred)
    true = torch.cat(true)
    exp_p = np.concatenate(exp_p)
    time_index = np.concatenate(time_index, axis=0)

    index = np.concatenate(index, axis=0)
    print('Index Max:', np.amax(index))

    split_date = '2022-12-31'
    split = deal_str_date(split_date)
    time_mask = time_index >= split

    mae = torch.abs(pred - true).numpy()
    mse = ((pred - true) ** 2).numpy()
    cmae1 = censor_mae(true, pred, 0.01, 0.01).numpy()
    cmse1 = censor_mse(true, pred, 0.01, 0.01).numpy()
    cmae2 = censor_mae(true, pred, 0.001, 0.001).numpy()
    cmse2 = censor_mse(true, pred, 0.001, 0.001).numpy()

    exp_mae = np.abs(exp_p - true.numpy())

    print(f'{seed}: MAE: {np.mean(mae)}, MSE: {np.mean(mse)},  CMAE1: {cmae1}, CMSE1: {cmse1},  CMAE2: {cmae2}, CMSE2: {cmse2}')

    binary_label = (true >= -1).float()  # (true >= -3).float() + (true >= -1.5).float()
    binary_pred = (pred >= -1).float()  # (pred >= -3).float() + (pred >= -1.5).float()
    cm = confusion_matrix(binary_label.numpy(), binary_pred.numpy())
    print(cm, accuracy_score(binary_label.numpy(), binary_pred.numpy()))

    if args.time_mask:
        mae = mae[time_mask]
        exp_mae = exp_mae[time_mask]
        index = index[time_mask]
        if out_type != 'time':
            exp_p = exp_p[time_mask]
            index_pred = index_pred[time_mask]
            index_input = index_input[time_mask]

    bin_edges = np.arange(0, 3650, 30)
    bin_centers = bin_edges[:-1] + 30 / 2

    errors = []
    variances = []
    exp_errors = []
    exp_variance = []
    num_samples = []
    for i in range(len(bin_centers)):
        # print(type(index), index.shape)
        bin_mask = (index >= bin_edges[i]) & (index < bin_edges[i + 1])

        bin_errors = mae[bin_mask]
        bin_mean_error = np.mean(bin_errors) if bin_errors.size else np.nan
        bin_variance = np.var(bin_errors) if bin_errors.size else np.nan
        errors.append(bin_mean_error)
        variances.append(bin_variance)

        bin_errors = exp_mae[bin_mask]
        bin_mean_error = np.nanmean(bin_errors) if bin_errors.size else np.nan
        bin_variance = np.nanvar(bin_errors) if bin_errors.size else np.nan
        exp_errors.append(bin_mean_error)
        exp_variance.append(bin_variance)

        num_samples.append(np.sum(bin_mask))

    for key, value in linage_result.items():
        linage_result[key] = np.array(sorted(value, key=lambda x: x[0])).T
    for key, value in linage_label.items():
        linage_label[key] = np.array(sorted(value, key=lambda x: x[0])).T
    for key, value in linage_label_smooth.items():
        linage_label_smooth[key] = np.array(sorted(value, key=lambda x: x[0])).T
    for key, value in linage_SNR.items():
        linage_SNR[key] = np.array(sorted(value, key=lambda x: x[0])).T
    for key, value in linage_numdata.items():
        linage_numdata[key] = np.array(sorted(value, key=lambda x: x[0])).T
    for key, value in linage_inputfreq.items():
        linage_inputfreq[key] = np.array(sorted(value, key=lambda x: x[0])).T
    for key, value in linage_error.items():
        linage_error[key] = np.array(sorted(value, key=lambda x: x[0])).T
    for key, value in linage_exp.items():
        linage_exp[key] = np.array(sorted(value, key=lambda x: x[0])).T

    return np.array(errors), np.array(variances), bin_centers, np.array(
        num_samples), pred, true, linage_result, linage_label, linage_label_smooth, linage_SNR, linage_numdata, linage_inputfreq, linage_error, linage_exp, np.array(
        exp_errors), np.array(exp_variance)


def test_future(args, x_test, nan_ind, test_name, seed, pop_list, save_path_list, ensemble=False, prints=False):
    batch_size = args.batch_size

    x_test = torch.tensor(x_test).float()

    nan_ind = torch.tensor(nan_ind).bool()

    if args.use_feature == 'all':
        day_ind = 1
        if args.used_model == 'transformer_token_encoder':
            day_ind = 5
    elif args.use_feature == 'all2':
        day_ind = 3
    else:
        print('The feature option has not days index')
        sys.exit()

    model_list = []
    for save_path in save_path_list:
        for p, ind in enumerate(pop_list):
            if 'reg_classify' in save_path:
                model = network.__dict__['transformer_reg_classify_encoder'](x_test.shape[-1], 1, args.dims, day=args.day,
                                                                             activation='gelu',
                                                                             norm=args.norm, dropout=args.dropout).to(
                    device)
            elif 'large' in save_path:
                model = network.__dict__['transformer_encoder_larger'](x_test.shape[-1], 1, [36], day=args.day, activation='gelu',
                                                                       norm=args.norm, dropout=args.dropout).to(device)
            elif 'combain' in save_path:
                model = network.__dict__[args.used_model](x_test.shape[-1], 1, args.dims, day=args.day, activation='gelu',
                                                          norm=args.norm, dropout=args.dropout, second=True).to(device)
            elif 'moe' in save_path:
                model = network.__dict__['moe'](x_test.shape[-1], 1, args.dims, day=args.day, activation='gelu',
                                                norm=args.norm, dropout=args.dropout, p=len(pop_list)).to(device)
            elif 'seq' in save_path:
                model = network.__dict__['transformer_seq_encoder'](x_test.shape[-1], 1, args.dims * 2, day=args.day, activation='gelu',
                                                                    norm=args.norm, dropout=args.dropout, p=len(pop_list)).to(device)
            else:
                model = network.__dict__[args.used_model](x_test.shape[-1], 1, args.dims, day=args.day, activation='gelu',
                                                          norm=args.norm, dropout=args.dropout).to(device)
            checkpoint = torch.load(save_path + f'_{p}.pth', map_location='cpu')
            print(save_path + f'_{p}.pth')
            model.load_state_dict(checkpoint['model'])
            model.eval()

            model_list.append(model)

    if ensemble:
        reg_result = ensamble_expReg(x_test, args.day, args.future)
        x_test = torch.cat([x_test, reg_result], dim=-1).float()

    dataset_valid = InverNetDataset_name3(x_test, nan_ind, test_name)

    valid_sampler = SequentialSampler(dataset_valid)

    test_loader = DataLoader(
        dataset_valid, batch_size=batch_size,
        sampler=valid_sampler, num_workers=12,
        pin_memory=True, collate_fn=default_collate)

    # Test the model
    linage_result = defaultdict(list)

    linage_numdata = defaultdict(list)
    linage_inputfreq = defaultdict(list)

    linage_exp = defaultdict(list)
    outlayers = []

    with torch.no_grad():
        pred = []

        exp_p = []
        index = []
        index_pred = []
        index_input = []
        time_index = []
        for images, nan, names in test_loader:
            images = images.to(device)
            nan = nan.to(device)

            if 'combain' not in save_path_list[0] and 'seq' not in save_path_list[0]:
                all_outputs = [model(images) for model in model_list]
                num_model = len(save_path_list)
            else:
                recent_outputs = [model(images) for model in model_list[len(pop_list):]]
                recent = [torch.clamp(out[:, 0].unsqueeze(1), max=0) for out in recent_outputs]
                recent = torch.stack(recent, dim=0)
                recent = torch.mean(recent, dim=0)  # batch, 1

                all_outputs = [model(images, recent) for model in model_list[:len(pop_list)]]
                num_model = 1

            outputs = [torch.clamp(out[:, 0].unsqueeze(1), max=0) for out in all_outputs]
            outputs = torch.stack(outputs, dim=0).reshape(num_model, len(pop_list), images.shape[0], 1)
            outputs = torch.mean(outputs, dim=1)  # number of model, batch, 1

            d = images.reshape([images.shape[0], args.day, -1]).detach().cpu().numpy()
            m = nan.reshape([images.shape[0], args.day, -1]).detach().cpu().numpy()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                valid_count, input_freq = compute_snr_and_valid_count_np_nolabel(np.log10(d[:, :, 0]), m[:, :, 0])

            exp_pred = []

            for q in range(d.shape[0]):
                m_nan = m[q, :, 0]
                x = d[q, :, day_ind]
                v = d[q, :, 0]

                # v = np.log10(v[m_nan])
                v = np.clip(v[m_nan], 1e-6, 1 - 1e-6)
                v = np.log(v / (1 - v))
                x = x[m_nan]

                # m_nan = np.isnan(v)
                # v = v[~m_nan]
                # x = x[~m_nan]

                if v.shape[0] < 2:
                    exp_pred.append(np.nan)
                else:
                    clf = LinearRegression()
                    clf.fit(x.reshape(-1, 1), v.reshape(-1, 1))

                    epred = clf.predict(d[q, -1, day_ind].reshape(1, 1) + args.future)
                    # epred = np.clip(epred, -4, 0)

                    epred = np.clip(epred, -15, 15)
                    epred = np.log10(1 / (1 + np.exp(-epred)))

                    exp_pred.append(epred[0, 0])

            exp_pred = np.stack(exp_pred)

            outputs = torch.mean(outputs, dim=0)

            date_index = images.reshape([images.shape[0], args.day, -1]).detach().cpu().numpy()[:, -1, day_ind] + args.future

            pred.append(outputs.detach().cpu().squeeze(-1))
            exp_p.append(exp_pred)
            time_index.append(date_index)

            index.append(date_index)

            for i, name in enumerate(names):
                linage_result[name].append([date_index[i], outputs[i].detach().cpu().squeeze().numpy()])

                linage_numdata[name].append([date_index[i] - args.future, valid_count[i]])
                linage_inputfreq[name].append([date_index[i] - args.future, input_freq[i]])

                if not np.isnan(exp_pred[i]):
                    linage_exp[name].append([date_index[i], exp_pred[i]])

    if prints and len(outlayers):
        print(set(outlayers))

    pred = torch.cat(pred)

    index = np.concatenate(index, axis=0)
    print('Index Max:', np.amax(index))

    for key, value in linage_result.items():
        linage_result[key] = np.array(sorted(value, key=lambda x: x[0])).T

    for key, value in linage_numdata.items():
        linage_numdata[key] = np.array(sorted(value, key=lambda x: x[0])).T
    for key, value in linage_inputfreq.items():
        linage_inputfreq[key] = np.array(sorted(value, key=lambda x: x[0])).T

    for key, value in linage_exp.items():
        linage_exp[key] = np.array(sorted(value, key=lambda x: x[0])).T

    return pred, linage_result, linage_numdata, linage_inputfreq, linage_exp


def trans_date(days_elapsed):
    start_date = datetime.datetime(2020, 1, 1)
    target_date = start_date + datetime.timedelta(days=days_elapsed)
    return target_date.strftime('%Y-%m-%d')


def main(args):
    print(args)

    score_version = 'Concern2'
    bar_dict = {'Concern1': [0.001, 0.029],
                'Concern2': [0.05, 0.1]}  # 0.1, 2;   0.5, 3,        0.001, 0.029,         0.047, 0.5
    sbar = bar_dict[score_version]

    step = 1
    ratio = 5
    binary = False
    random_split = True
    cv = args.cv

    consecutive_days = 3
    max_day = 3650

    input_feature = 3
    out_feature = 3

    exp_days = args.day  # 14
    max_day2 = args.max_day2

    interp = args.interp
    token = -4

    skip_days = args.skip_days

    save_path = f'{args.output_dir}/{args.used_model + args.ckpt_name}/label_{args.label_type}/test_multi_model_{args.data_phase}/'
    try:
        os.makedirs(save_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    colors = ["#ffa510", "#ffbd66", "#41b7ac", "#0c84c6", "#2455a4", "#002c53"] * 7
    line_format = ['-o'] * 7

    checkpoint_list = []
    name_list = []
    used_model_list = []
    dim_list = []

    def process_checkpoints(ckpt_names, seeds, ums, days, label_type=args.label_type):
        batch_ckpt, batch_name, batch_um = [], [], []
        for ckpt_name, s, um, d in zip(ckpt_names, seeds, ums, days):
            checkpoint_path = f'{args.ckpt_dir}/{um + ckpt_name}/label_{label_type}'
            batch_ckpt.append(checkpoint_path + f'/seed{s}/checkpoint_{42}_{d}_all2_seed{s}')
            batch_name.append(ckpt_name)
            batch_um.append(um)
        return batch_ckpt, batch_um[0]

    future_list = [14, 21, 28, 35, 42, 60]
    seeds = [[0], [1, 0], [7, 0], [8, 0], [4, 0], [4, 0]]
    ckpt_names_list = [['final_drop_token']] + [['final_drop_token_combain', 'final_drop_token']] * 5
    ums_list = [['transformer_encoder2']] + [['transformer_encoder2'] * 2] * 5
    dims = [[8], [8], [8], [8], [8], [8]]
    for ckpt_names, day, seed, ums, dim in zip(ckpt_names_list, future_list, seeds, ums_list, dims):
        input_day = [14] if day == 14 else [day, 14]
        ckpt, um = process_checkpoints(ckpt_names, seed, ums, input_day)
        checkpoint_list.append(ckpt)
        name_list.append('Transformer' + f'_{day}D')
        used_model_list.append(um)
        dim_list.append(dim)
    save_file_name = args.save_file_name

    df = pd.read_csv('Number_totalSeqs_new.csv')
    filtered_df = df[df['totalSeqs'] > 0]
    country_list = filtered_df['Country'].tolist()

    remove_isolate = False
    consecutive_days = 0
    if args.train_process:
        remove_isolate = True
        consecutive_days = 3

    large_pango = ['B.1', 'BA.1.1.2', 'B.1.177.10', 'AY.103.1', 'XBB.1.9.1', 'BQ.1.1.1', 'B.1.429', 'B.1.469', 'AY.4.2.1', 'B.1.1.13', 'CH.1.1',
                   'B.1.493', 'CH.1.1.1', 'B.1.37', 'BA.5.2.1', 'BA.5.2.13', 'B.40', 'B.1.1.178', 'XBB.1.5', 'B.1.1.70', 'B.1.466.1', 'B.1.1.303',
                   'B.1.596', 'XBB.1.1', 'B.1.444', 'BA.5.1.1', 'BA.1.1.1', 'B.1.105', 'AY.3.1', 'B.1.595.3', 'B.1.1.240', 'B.45', 'B.1.1.45',
                   'BA.2.1', 'B.1.427', 'B.1.1.406', 'B.1.565', 'B.1.243.1', 'BA.5.2.10', 'B.1.111', 'XBB.1.9', 'CH.1.1.4', 'B.57', 'B.1.206',
                   'B.1.1.15', 'B.1.1.256', 'CH.1.1.3', 'B.1.1.234', 'AY.25.1', 'B.1.1.8', 'B.1.391', 'B.1.1.408', 'XBB.1.6', 'BA.4.1', 'B.1.1',
                   'B.1.1.371', 'BA.1.15.1', 'AY.4.1', 'XBB.2', 'B.1.1.37', 'B.1.306', 'B.1.1.529', 'B.1.595.1', 'B.1.177.74', 'B.1.1.1',
                   'B.1.539', 'A.2', 'AY.45', 'BA.1.18', 'B.1.1.304', 'BA.5.5.1', 'B.1.1.71', 'B.1.1.200', 'B.1.1.29', 'BA.2.13', 'A.1', 'BA.1.1',
                   'B.1.429.1', 'CH.1.1.2', 'B.1.1.218', 'B.1.1.10', 'AK.1', 'B.1.241']

    no_data_lsit = []
    missing_file_list, missing_file_list2 = [], []

    pred_dict_list = [{} for _ in range(len(checkpoint_list))]
    true_dict_list = [{} for _ in range(len(checkpoint_list))]
    true_smooth_dict_list = [{} for _ in range(len(checkpoint_list))]
    snr_dict_list = [{} for _ in range(len(checkpoint_list))]
    datapoint_dict_list = [{} for _ in range(len(checkpoint_list))]
    inputfreq_dict_list = [{} for _ in range(len(checkpoint_list))]
    error_dict_list = [{} for _ in range(len(checkpoint_list))]
    exp_dict_list = [{} for _ in range(len(checkpoint_list))]
    pred_future_dict_list = [{} for _ in range(len(checkpoint_list))]
    datapoint_future_dict_list = [{} for _ in range(len(checkpoint_list))]
    inputfreq_future_dict_list = [{} for _ in range(len(checkpoint_list))]
    exp_dict_future_list = [{} for _ in range(len(checkpoint_list))]
    smoothed, raw = {}, {}

    print(f'There are {len(country_list)} countries in total.')

    for i, country in enumerate(country_list):
        print('""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'
              '""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""')
        print("*" * 10 + country + '*' * 10)

        future_splits = '2024-02-28'  # '2023-12-31'
        train_val_splits = '2022-12-31' if country in ['GBR', 'USA'] else '2020-12-31'  # '2023-03-31'

        file_path = os.path.join(f'{args.anno_path}/collect_date_{args.dataset_version}', 'freqs_bySubDate_' + country + '.csv')
        file_path2 = os.path.join(f'{args.anno_path}/collect_date_{args.dataset_version}', 'freqs_final_' + country + '.csv')

        print(i, train_val_splits)
        print(file_path)
        print(file_path2)

        if not os.path.exists(file_path):
            missing_file_list.append(country)
            continue
        if not os.path.exists(file_path2):
            missing_file_list2.append(country)
            continue

        start_date_train, original_dict_train, dict_test, df_sort = load_data_new_time(
            f'{args.anno_path}/collect_date_{args.dataset_version}', country, suffix='',
            consecutive_days=consecutive_days, new_split=args.new_split, split_date=train_val_splits)

        b1, i1 = clean_data(original_dict_train, interp=interp, token=token, nan_indicat=True)
        b5, i5 = clean_data(dict_test, interp=interp, token=token, nan_indicat=True)

        if args.data_phase == 'test':
            c_data = b5
            c_ind = i5
        elif args.data_phase == 'train':
            c_data = b1
            c_ind = i1

        try:
            start_date_test, raw_dict, original_dict_test, log_dict_test, interpolated_dict_test, mask_dict_test, smoothed_dict_test, df_sort_test, noise_level, noise_ind, noise_ind2 = load_label_smooth(
                f'{args.anno_path}/collect_date_{args.dataset_version}', country, output_noise_level=True, remove_isolate=remove_isolate)
        except:
            no_data_lsit.append(country)
            continue

        original_dict_test = clean_label(original_dict_test, interp=interp, token=10 ** token)

        print(start_date_train, start_date_test)
        print(len(original_dict_train), len(dict_test), len(smoothed_dict_test))

        if args.label_type in ['original', 'log_originallabel', 'smooth_originallabel']:
            b7 = original_dict_test
        elif args.label_type in ['log', 'smooth_loglabel']:
            b7 = interpolated_dict_test
        elif args.label_type in ['raw', 'original_raw', 'log_raw', 'smooth_raw']:
            b7 = raw_dict
        elif args.label_type == 'smooth':
            b7 = smoothed_dict_test
        else:
            b7 = interpolated_dict_test

        print(len(c_data), len(b7))

        dict_var = read_concern('related_files/concern-GBR-USA.csv', name=score_version)

        if args.use_feature == 'all':
            position = True
            features = [input_feature, 0, 1, 2, 4, 5]  # 0: Abosulute day, 1: totalSeqs, 2: VarSeqs, 3: VarFreq, 4/5: country, 6(True): relevant day
            if args.used_model == 'transformer_token_encoder':
                features = [input_feature, 1, 2, 4, 5, 0]
        elif args.use_feature == 'all2':
            position = True
            features = [input_feature, 1, 2, 0]
        elif args.use_feature == 'freq':
            position = False
            features = [input_feature]
        elif args.use_feature == 'freq+pc1':
            position = False
            features = [input_feature, 4, 5]
        else:
            print('Did not write the feature option')
            sys.exit()

        pop_list = [(r, r + ratio - 1) for r in range(ratio)] if cv else [(ratio - 1, ratio * 2 - 2)]
        plot_ind = 0
        for checkpoint, line_name, line_color, lf, um, dim in zip(checkpoint_list, name_list, colors, line_format, used_model_list, dim_list):
            day = int(checkpoint[0].split('_')[-4])
            future = int(checkpoint[0].split('_')[-3])
            if '_seq' in line_name:
                future = int(line_name.split('_')[-1])
            args.day = day
            args.future = future
            args.used_model = um
            args.dims = dim
            # if '2L' in checkpoint[0]:
            #     args.dims = [8, 8]
            # elif '3L' in checkpoint[0]:
            #     args.dims = [8, 8, 8]
            # else:
            #     args.dims = [8]
            print('\n' * 2)
            print(line_name)
            print('&&&&&&&&&&', plot_ind)
            print('Day:', args.day, 'Future:', args.future, 'Dims:', args.dims)
            print(checkpoint)

            random.seed(1)
            train_set3, train_label3, train_label4, train_label5, train_label3_smooth, linage3, exp_set3, exp_linage3, exp_label3, samll_label3, less_label3, lag_label3 = \
                generate_cross_validation_one_country_new(c_data, b7, dict_var, sbar, max_day=max_day2, days=day,
                                                          exp_days=exp_days,
                                                          future=future, ratio=1, position=position, features=features,
                                                          input_feature=input_feature, out_feature=out_feature,
                                                          random_split=random_split, token=token,
                                                          prefix_padding=args.prefix_padding,
                                                          smoothed_label=smoothed_dict_test, nan_ind=c_ind, drop_rule=args.drop_rule,
                                                          split_date=future_splits)

            if train_set3 is None and ('freqs_lowNumCountries' in args.dataset_version or 'v4' in args.dataset_version):
                no_data_lsit.append(country)
                continue

            seed = int(checkpoint[0].split('_')[-1][4:])
            torch.manual_seed(seed)

            """## GBR"""

            test_set, test_label, test_label_smooth = np.concatenate(train_set3, axis=0), np.concatenate(train_label3, axis=0), np.concatenate(
                train_label3_smooth, axis=0)
            smooth_label = np.concatenate(train_label4, axis=0)
            nan_ind = np.concatenate(train_label5, axis=0)
            test_name = linage3[0]
            errors, variances, bin_centers, num_samples, pred, true, linage_result, linage_label, linage_label_smooth, linage_SNR, linage_numdata, linage_inputfreq, linage_error, linage_exp, exp_error, exp_var = test(
                args, test_set,
                test_label, test_label_smooth,
                smooth_label, nan_ind,
                test_name, seed,
                pop_list,
                checkpoint,
                ensemble=False,
                prints=False, out_type='time')

            print(country, 'Day:', day, 'Future:', future, num_samples)

            pred_dict_list[plot_ind].update(linage_result)
            true_dict_list[plot_ind].update(linage_label)
            true_smooth_dict_list[plot_ind].update(linage_label_smooth)
            snr_dict_list[plot_ind].update(linage_SNR)
            datapoint_dict_list[plot_ind].update(linage_numdata)
            inputfreq_dict_list[plot_ind].update(linage_inputfreq)
            error_dict_list[plot_ind].update(linage_error)
            exp_dict_list[plot_ind].update(linage_exp)

            smoothed.update(smoothed_dict_test)
            raw.update(raw_dict)

            if args.test_future:
                print('"""""Future Start"""""')

                random.seed(1)
                train_set_future, train_nan_future, linage_future, exp_set_future, flat_linage_future, less_label_future, lag_label_future = \
                    generate_cross_validation_one_country_future(c_data, b7, dict_var, sbar, max_day=max_day2, days=day,
                                                                 exp_days=exp_days,
                                                                 future=future, ratio=1, position=position, features=features,
                                                                 input_feature=input_feature, out_feature=out_feature,
                                                                 random_split=random_split, token=token,
                                                                 prefix_padding=args.prefix_padding,
                                                                 smoothed_label=smoothed_dict_test, nan_ind=c_ind, drop_rule=args.drop_rule,
                                                                 split_date=future_splits)

                seed = int(checkpoint[0].split('_')[-1][4:])
                torch.manual_seed(seed)

                """## GBR"""

                test_set = np.concatenate(train_set_future, axis=0)
                nan_ind = np.concatenate(train_nan_future, axis=0)
                test_name = linage_future[0]
                pred, linage_result, linage_numdata, linage_inputfreq, linage_exp = test_future(args, test_set, nan_ind, test_name, seed, pop_list,
                                                                                                checkpoint, ensemble=False, prints=False)

                pred_future_dict_list[plot_ind].update(linage_result)
                datapoint_future_dict_list[plot_ind].update(linage_numdata)
                inputfreq_future_dict_list[plot_ind].update(linage_inputfreq)
                exp_dict_future_list[plot_ind].update(linage_exp)

                print('"""""Future End"""""')

            plot_ind += 1

            print()

        print('""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'
              '""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""\n\n')

    print(no_data_lsit)
    print(missing_file_list)
    print(missing_file_list2)
    phase = args.data_phase == 'test'
    save_all(pred_dict_list, true_dict_list, true_smooth_dict_list, exp_dict_list, raw, smoothed, snr_dict_list, datapoint_dict_list,
             inputfreq_dict_list, error_dict_list, large_pango, name_list, colors, line_format, save_path, phase,
             future_list, pred_future_dict_list, datapoint_future_dict_list, inputfreq_future_dict_list, exp_dict_future_list,
             file_name=save_file_name)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
