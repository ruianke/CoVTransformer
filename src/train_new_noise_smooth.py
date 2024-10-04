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

import torch
from torch import nn
from torch.utils.data.dataloader import default_collate
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler, WeightedRandomSampler


import network
from utilities import * # analysis:ignore
from network import * # analysis:ignore
import errno

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--num_epochs', default=500, type=int)
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float, help='learning_rate')
    parser.add_argument('-wd', '--weight_decay', default=1e-4, type=float, help='learning_rate')
    parser.add_argument('--dropout', default=0., type=float, help='learning_rate')

    parser.add_argument('--norm', default=False, type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument('--lr_scheduler', default=False, type=bool, action=argparse.BooleanOptionalAction)

    parser.add_argument('--day', default=42, type=int)
    parser.add_argument('--skip_days', default=14, type=int)
    parser.add_argument('--future', default=14, type=int)
    parser.add_argument('--max_day2', default=3650, type=int)

    parser.add_argument('--use_feature', default='all2', type=str)
    parser.add_argument('--used_model', default='transformer_encoder2', type=str)
    parser.add_argument('--loss', default='weighted_exp', type=str)
    parser.add_argument('--loss_list', nargs='+', default=[], type=str)
    parser.add_argument('--para_noise_level', default=1e-4, type=float)
    parser.add_argument('--noise_level', default=-0.1, type=float)
    parser.add_argument('--label_type', default='smooth_raw', type=str)

    parser.add_argument('--dims', nargs='+', default=[8], type=int)
    parser.add_argument('--cv', default=True, type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument('--high_only', default=False, type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument('--resample', default=False, type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument('--seq_out', default=False, type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument('--drop_rule', default=-1, type=int)
    parser.add_argument('--new_split', default=False, type=bool, action=argparse.BooleanOptionalAction)

    parser.add_argument('--low_model', default=0, type=int)
    parser.add_argument('--high_model', default=0, type=int)

    parser.add_argument('--seed_list', nargs='+', default=[], type=int)
    parser.add_argument('--save_model', default=True, type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument('--ckpt_name', default='', type=str)
    parser.add_argument('--test_on_new', default=True, type=bool, action=argparse.BooleanOptionalAction)

    parser.add_argument('--prefix_padding', default=True, type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument('--interp', default=False, type=bool, action=argparse.BooleanOptionalAction)

    parser.add_argument('--suffix', default='_90days', type=str)
    parser.add_argument('--dataset_version', default='v2', type=str)

    parser.add_argument('--ckpt_dir', default='./ckpt',
                        help='path where to save and load ckpt')
    parser.add_argument('--anno-path', default='./dataset',
                        help='path where to load data')

    return parser


def make_balanced_sampler(targets, num_bins):
    # Bin the target variable
    bins = np.linspace(targets.min()-0.01, targets.max()+0.01, num_bins)
    y_binned = np.digitize(targets, bins)

    # Count the number of instances in each bin
    bin_counts = np.bincount(y_binned)

    # Define weights as the inverse of the bin counts
    weights = 1. / bin_counts[1:]
    weights = np.insert(weights, 0, 0)

    print(targets.min(), targets.max(), targets.shape)
    print(bins)
    print(f'Bins: {bin_counts}')
    print(f'Weights: {weights}')

    # Assign each instance the weight of its bin
    sample_weights = weights[y_binned]

    # Use the weights to create a WeightedRandomSampler
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    return sampler


def train(args, seed, x_train, y_train, x_test, y_test, test_name, x_test2, y_test2, test_name2, dims=None,
                  ensemble=False, prints=False, low_model_path=None, high_model_path=None, num_p=5):
    if dims is None:
        dims = [8]
    # Hyper-parameters
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    x_train = torch.tensor(x_train).float()
    y_train = torch.tensor(y_train).float()
    x_test = torch.tensor(x_test).float()
    y_test = torch.tensor(y_test).float()
    x_test2 = torch.tensor(x_test2).float()
    y_test2 = torch.tensor(y_test2).float()

    if args.label_type == 'original':
        y_train = torch.log10(y_train)
        y_test = torch.log10(y_test)
        y_test2 = torch.log10(y_test2)
    elif args.label_type == 'raw':
        y_train = torch.log10(y_train)
        y_test = torch.log10(y_test)
        y_test2 = torch.log10(y_test2)
    elif args.label_type == 'original_raw':
        y_train = torch.log10(y_train)
        y_test = torch.log10(y_test)
        y_test2 = torch.log10(y_test2)
    elif args.label_type == 'log_originallabel':
        y_test = torch.log10(y_test)
        y_test2 = torch.log10(y_test2)
    elif args.label_type == 'smooth_originallabel':
        y_test = torch.log10(y_test)
        y_test2 = torch.log10(y_test2)
    elif args.label_type == 'log_raw':
        y_test = torch.log10(y_test)
        y_test2 = torch.log10(y_test2)
    elif args.label_type == 'smooth_raw':
        y_test = torch.log10(y_test)
        y_test2 = torch.log10(y_test2)


    if ensemble:
        reg_result = ensamble_expReg(x_train, args.day, args.future)
        x_train = torch.cat([x_train, reg_result], dim=-1).float()

        reg_result = ensamble_expReg(x_test, args.day, args.future)
        x_test = torch.cat([x_test, reg_result], dim=-1).float()

        reg_result = ensamble_expReg(x_test2, args.day, args.future)
        x_test2 = torch.cat([x_test2, reg_result], dim=-1).float()

    dataset_train = InverNetDataset(x_train, y_train)
    dataset_valid = InverNetDataset_name(x_test, y_test, test_name)
    dataset_valid2 = InverNetDataset_name(x_test2, y_test2, test_name2)

    # print('Creating data loaders')
    if not args.resample:
        train_sampler = RandomSampler(dataset_train)
    else:
        train_sampler = make_balanced_sampler(y_train, num_bins=5)
    valid_sampler = SequentialSampler(dataset_valid)
    valid_sampler2 = SequentialSampler(dataset_valid2)

    train_loader = DataLoader(
        dataset_train, batch_size=batch_size,
        sampler=train_sampler, num_workers=12,
        pin_memory=True, drop_last=False, collate_fn=default_collate)

    test_loader = DataLoader(
        dataset_valid, batch_size=batch_size,
        sampler=valid_sampler, num_workers=12,
        pin_memory=True, collate_fn=default_collate)
    test_loader2 = DataLoader(
        dataset_valid2, batch_size=batch_size,
        sampler=valid_sampler2, num_workers=12,
        pin_memory=True, collate_fn=default_collate)

    model = network.__dict__[args.used_model](x_train.shape[-1], 1, dims, day=args.day, activation='gelu',
                                              norm=args.norm,  dropout=args.dropout, p=num_p).to(device)

    if 'moe' in args.used_model and low_model_path is not None and high_model_path is not None:
        network.load_pretrained_models_moe(model, low_model_path, high_model_path)

        # Freeze the parameters of the recent_model
        for pname, param in model.named_parameters():
            if 'low_model' in pname or 'high_model' in pname:
                param.requires_grad = False

    # Loss and optimizer
    if args.loss == 'combain':
        criterion = combain_loss
    elif args.loss == 'l1':
        criterion = nn.L1Loss()
    elif args.loss == 'l2':
        criterion = nn.MSELoss()
    elif args.loss == 'huber':
        criterion = nn.HuberLoss()
    elif args.loss == 'quantile':
        criterion = quantile_loss([0.3])
    elif args.loss == 'weighted_quad':
        criterion = weighted_combain_loss_quad
    elif args.loss == 'weighted_mask':
        criterion = weighted_combain_loss_mask
    elif args.loss == 'weighted_tanh':
        criterion = weighted_combain_loss_tanh
    elif args.loss == 'weighted_exp':
        criterion = weighted_combain_loss_exp
    elif args.loss == 'weighted_exp2':
        criterion = weighted_combain_loss_exp2
    elif args.loss == 'weighted_exp3':
        criterion = weighted_combain_loss_exp3
    elif args.loss == 'censored':
        criterion = combain_censor_loss
    elif args.loss == 'weighted_tanh2':
        criterion = weighted_combain_loss_tanh2
    elif args.loss == 'weighted_censored_tanh':
        criterion = weighted_combain_loss_censored_tanh
    elif args.loss == 'piece_linear':
        criterion = weighted_combain_loss_piece_linear
    else:
        criterion = combain_loss

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
    if args.lr_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, eta_min=0, T_0=5, T_mult=2)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
            
            if args.high_only:
                mask = (labels>= -1.5).float()
                labels = labels * mask + (labels * 0 - 3) * (1-mask)

            noise_list = []
            for param in model.parameters():
                noise = torch.randn(param.data.size()).to(device) * args.para_noise_level
                param.data += noise
                noise_list.append(noise)

            # Forward pass
            outputs = model(images)
            if 'reg_classify' in args.used_model:
                loss_reg = criterion(labels.squeeze(), outputs[:,0].squeeze())
                loss_classify = classify_high_freq_loss(labels.squeeze(), outputs[:,1].squeeze())
                loss = (loss_reg+loss_classify)/2.0
            else:
                loss = criterion(labels.squeeze(), outputs.squeeze())  # , c2=0.001

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for param in model.parameters():
                param.data -= noise_list[0].to(device)
                noise_list = noise_list[1:]

            if args.lr_scheduler:
                lr_scheduler.step(epoch + i / total_step)

            if (i + 1) % 2000 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    # Test the model
    linage_result = defaultdict(list)
    linage_label = defaultdict(list)
    outlayers = []
    model.eval()
    with torch.no_grad():
        pred = []
        true = []
        for images, labels, names in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            outputs[:,0] = torch.clamp(outputs[:,0], max=0)

            if 'reg_classify' in args.used_model:
                pred.append(outputs[:,0].detach().cpu().squeeze().reshape([images.shape[0]]))
            else:
                pred.append(outputs.detach().cpu().squeeze().reshape([images.shape[0], -1]).squeeze())
            true.append(labels.detach().cpu().squeeze().reshape([images.shape[0], -1]).squeeze())

            for i, name in enumerate(names):
                linage_result[name].append(outputs[i].detach().cpu().squeeze().numpy())
                linage_label[name].append(labels[i].detach().cpu().squeeze().numpy())

                if prints and (outputs[i] - labels[i]) ** 2 >= 0.25:
                    outlayers.append(name)

    if prints and len(outlayers):
        print(set(outlayers))

    pred = torch.cat(pred)
    true = torch.cat(true)

    cmae = censor_mae(true, pred, 0.01, 0.01)
    cmse = censor_mse(true, pred, 0.01, 0.01)
    mae = torch.mean(torch.abs(pred - true))
    mse = torch.mean((pred - true) ** 2)

    print(f'{seed}: MAE: {mae}, MSE: {mse}, CMAE: {cmae}, CMSE: {cmse}')

    # Test the model
    linage_result = defaultdict(list)
    linage_label = defaultdict(list)
    outlayers = []
    model.eval()
    with torch.no_grad():
        pred = []
        true = []
        for images, labels, names in test_loader2:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            outputs[:,0] = torch.clamp(outputs[:,0], max=0)

            if 'reg_classify' in args.used_model:
                pred.append(outputs[:,0].detach().cpu().squeeze().reshape([images.shape[0]]))
            else:
                pred.append(outputs.detach().cpu().reshape([images.shape[0], -1]).squeeze())
            true.append(labels.detach().cpu().reshape([images.shape[0], -1]).squeeze())

            for i, name in enumerate(names):
                linage_result[name].append(outputs[i].detach().cpu().squeeze().numpy())
                linage_label[name].append(labels[i].detach().cpu().squeeze().numpy())

                if prints and (outputs[i] - labels[i]) ** 2 >= 0.25:
                    outlayers.append(name)

    if prints and len(outlayers):
        print(set(outlayers))

    pred = torch.cat(pred)
    true = torch.cat(true)

    cmae2 = censor_mae(true, pred, 0.01, 0.01)
    cmse2 = censor_mse(true, pred, 0.01, 0.01)
    mae2 = torch.mean(torch.abs(pred - true))
    mse2 = torch.mean((pred - true) ** 2)

    print(f'{seed}: MAE: {mae2}, MSE: {mse2}, CMAE: {cmae2}, CMSE: {cmse2}')

    return mae, mae2, mse, mse2, cmae, cmae2, cmse, cmse2, model


def exp_test(b1, b2, b3, b4, dict_var, sbar, skip_days,
             input_feature, output_feature, step, binary, position, random_split,
             future, features, max_day2, exp_days, ratio=5, token=-4, prefix_padding=True):
    position = False
    features = [input_feature]

    position = False
    features = [input_feature]

    random.seed(1)
    GBR_train_set3, GBR_train_label3, _, GBR_exp_set, GBR_linage3, GBR_exp_label, GBR_samll_label3, GBR_less_label3, GBR_lag_label3, GBR_num_data = \
        generate_cross_validation_one_country_exp(b1, b3, dict_var, sbar, max_day=max_day2, skip_days=skip_days,
                                                  exp_days=exp_days,
                                                  future=future, ratio=ratio, position=position, features=features,
                                                  input_feature=input_feature, out_feature=output_feature,
                                                  random_split=random_split, token=token, prefix_padding=prefix_padding)
    USA_train_set3, USA_train_label3, _, USA_exp_set, USA_linage3, USA_exp_label, USA_samll_label3, USA_less_label3, USA_lag_label3, USA_num_data = \
        generate_cross_validation_one_country_exp(b2, b4, dict_var, sbar, max_day=max_day2, skip_days=skip_days,
                                                  exp_days=exp_days,
                                                  future=future, ratio=ratio, position=position, features=features,
                                                  input_feature=input_feature, out_feature=output_feature,
                                                  random_split=random_split, token=token, prefix_padding=prefix_padding)
    """### exp"""

    exp_mae, exp_mse, count = 0, 0, 0
    exp_cmae, exp_cmse = 0, 0

    pred_list, label_list = [], []
    label_exp_dict = defaultdict(list)
    pred_exp_dict = defaultdict(list)

    for i in range(len(GBR_exp_set)):
        d = np.log10(GBR_exp_set[i])
        l = np.log10(GBR_exp_label[i])
        n = GBR_linage3[i][0]

        x = np.arange(d.shape[0])

        clf = LinearRegression()
        # clf = Ridge(alpha=1.0)

        clf.fit(x.reshape(-1, 1), d.reshape(-1, 1))

        pred = clf.predict(np.array(d.shape[0] + future).reshape(-1, 1))
        pred = np.minimum(pred, 0)

        exp_cmae += censor_mae(torch.tensor(pred), torch.tensor(l))
        exp_cmse += censor_mse(torch.tensor(pred), torch.tensor(l))
        exp_mae += torch.mean(torch.abs(torch.tensor(pred) - torch.tensor(l)))
        exp_mse += torch.mean((torch.tensor(pred) - torch.tensor(l)) ** 2)
        count += 1

        pred_list.append(pred)
        label_list.append(l)

        label_exp_dict[n].append(l)
        pred_exp_dict[n].append(pred[0][0])

    GBR_label_list_exp = np.stack(label_list)
    GBR_pred_list_exp = np.stack(pred_list)

    print(f'MAE: {exp_mae / count}, MSE: {exp_mse / count}, CMAE: {exp_cmae / count}, CMSE: {exp_cmse / count}', count)

    exp_mae, exp_mse, count = 0, 0, 0
    exp_cmae, exp_cmse = 0, 0

    pred_list, label_list = [], []

    for i in range(len(USA_exp_set)):
        d = np.log10(USA_exp_set[i])
        l = np.log10(USA_exp_label[i])
        n = USA_linage3[i][0]

        x = np.arange(d.shape[0])

        clf = LinearRegression()
        # clf = Ridge(alpha=1.0)

        clf.fit(x.reshape(-1, 1), d.reshape(-1, 1))

        pred = clf.predict(np.array(d.shape[0] + future).reshape(-1, 1))
        if input_feature == 1:
            pred = np.minimum(pred, 0)
        elif input_feature == 4:
            pred = np.maximum(np.minimum(pred, 6), 0)

        exp_cmae += censor_mae(torch.tensor(pred), torch.tensor(l))
        exp_cmse += censor_mse(torch.tensor(pred), torch.tensor(l))
        exp_mae += torch.mean(torch.abs(torch.tensor(pred) - torch.tensor(l)))
        exp_mse += torch.mean((torch.tensor(pred) - torch.tensor(l)) ** 2)
        count += 1

        pred_list.append(pred)
        label_list.append(l)

        label_exp_dict[n].append(l)
        pred_exp_dict[n].append(pred[0][0])

    USA_label_list_exp = np.stack(label_list)
    USA_pred_list_exp = np.stack(pred_list)

    for key, value in pred_exp_dict.items():
        pred_exp_dict[key] = np.stack(value)
    for key, value in label_exp_dict.items():
        label_exp_dict[key] = np.stack(value)

    print(f'MAE: {exp_mae / count}, MSE: {exp_mse / count}, CMAE: {exp_cmae / count}, CMSE: {exp_cmse / count}', count)

    return GBR_label_list_exp, GBR_pred_list_exp, USA_label_list_exp, USA_pred_list_exp


def test(args, model, x_test, y_test, test_name, ensemble=False, prints=False):
    batch_size = args.batch_size

    x_test = torch.tensor(x_test).float()
    y_test = torch.tensor(y_test).float()

    if args.label_type == 'original':
        y_test = torch.log10(y_test)
    elif args.label_type == 'log_originallabel':
        y_test = torch.log10(y_test)
    elif args.label_type == 'smooth_originallabel':
        y_test = torch.log10(y_test)
    elif args.label_type == 'log_raw':
        y_test = torch.log10(y_test)
    elif args.label_type == 'smooth_raw':
        y_test = torch.log10(y_test)
    elif args.label_type == 'raw':
        y_test = torch.log10(y_test)
    elif args.label_type == 'original_raw':
        y_test = torch.log10(y_test)

    if ensemble:
        reg_result = ensamble_expReg(x_test, args.day, args.future)
        x_test = torch.cat([x_test, reg_result], dim=-1).float()

    dataset_valid = InverNetDataset_name(x_test, y_test, test_name)

    valid_sampler = SequentialSampler(dataset_valid)

    test_loader = DataLoader(
        dataset_valid, batch_size=batch_size,
        sampler=valid_sampler, num_workers=12,
        pin_memory=True, collate_fn=default_collate)

    # Test the model
    linage_result = defaultdict(list)
    linage_label = defaultdict(list)
    outlayers = []
    model.eval()
    with torch.no_grad():
        pred = []
        true = []
        for images, labels, names in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            outputs[:,0] = torch.clamp(outputs[:,0], max=0)

            if 'reg_classify' in args.used_model:
                pred.append(outputs[:,0].detach().cpu().squeeze().reshape([images.shape[0]]))
            else:
                pred.append(outputs.detach().cpu().reshape([images.shape[0], -1]).squeeze())
            true.append(labels.detach().cpu().reshape([images.shape[0], -1]).squeeze())

            for i, name in enumerate(names):
                linage_result[name].append(outputs[i].detach().cpu().squeeze().numpy())
                linage_label[name].append(labels[i].detach().cpu().squeeze().numpy())

                if prints and (outputs[i] - labels[i]) ** 2 >= 0.25:
                    outlayers.append(name)

    if prints and len(outlayers):
        print(set(outlayers))

    pred = torch.cat(pred)
    true = torch.cat(true)

    cmae = censor_mae(true, pred, 0.01, 0.01)
    cmse = censor_mse(true, pred, 0.01, 0.01)
    mae = torch.mean(torch.abs(pred - true))
    mse = torch.mean((pred - true) ** 2)

    for key, value in linage_result.items():
        linage_result[key] = np.stack(value)
    for key, value in linage_label.items():
        linage_label[key] = np.stack(value)

    print(f'New Data: MAE: {mae}, MSE: {mse}, CMAE: {cmae}, CMSE: {cmse}')

    return true, pred, mae, mse, cmae, cmse, model, linage_result, linage_label


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

    day = args.day
    exp_days = args.day  # 14
    future = args.future
    max_day2 = args.max_day2

    interp = args.interp
    token = -4

    skip_days = args.skip_days

    GBR_start_date_train, GBR_original_dict_train, GBR_dict_test, GBR_df_sort = load_data_new_time(f'{args.anno_path}/collect_date_{args.dataset_version}', 'GBR', suffix=args.suffix,
                               consecutive_days=consecutive_days, new_split=args.new_split)
    USA_start_date_train, USA_original_dict_train, USA_dict_test, USA_df_sort  = load_data_new_time(f'{args.anno_path}/collect_date_{args.dataset_version}', 'USA', suffix=args.suffix,
                               consecutive_days=consecutive_days, new_split=args.new_split)

    b1, i1 = clean_data(GBR_original_dict_train, interp=interp, token=token, nan_indicat=True)
    b2, i2 = clean_data(USA_original_dict_train, interp=interp, token=token, nan_indicat=True)
    b5, i5 = clean_data(GBR_dict_test, interp=interp, token=token, nan_indicat=True)
    b6, i6 = clean_data(USA_dict_test, interp=interp, token=token, nan_indicat=True)

    GBR_start_date_test, GBR_raw_dict, GBR_original_dict_test, GBR_log_dict_test, GBR_interpolated_dict_test, GBR_mask_dict_test, GBR_smoothed_dict_test, GBR_df_sort_test = load_label_smooth(
        f'{args.anno_path}/collect_date_{args.dataset_version}', 'GBR')
    USA_start_date_test, USA_raw_dict, USA_original_dict_test, USA_log_dict_test, USA_interpolated_dict_test, USA_mask_dict_test, USA_smoothed_dict_test, USA_df_sort_test = load_label_smooth(
        f'{args.anno_path}/collect_date_{args.dataset_version}', 'USA')

    GBR_original_dict_test = clean_label(GBR_original_dict_test, interp=interp, token=10 ** token)
    USA_original_dict_test = clean_label(USA_original_dict_test, interp=interp, token=10 ** token)

    print(GBR_start_date_train, USA_start_date_train, GBR_start_date_test, USA_start_date_test)
    print(len(GBR_original_dict_train), len(USA_original_dict_train), len(GBR_dict_test),
          len(USA_dict_test), len(GBR_smoothed_dict_test), len(USA_smoothed_dict_test))

    if args.label_type == 'original':
        b3, b4 = GBR_original_dict_test, USA_original_dict_test
        b3_val, b4_val = GBR_original_dict_test, USA_original_dict_test
        b7, b8 = GBR_original_dict_test, USA_original_dict_test
    elif args.label_type == 'log':
        b3, b4 = GBR_interpolated_dict_test, USA_interpolated_dict_test
        b3_val, b4_val = GBR_interpolated_dict_test, USA_interpolated_dict_test
        b7, b8 = GBR_interpolated_dict_test, USA_interpolated_dict_test
    elif args.label_type == 'log_originallabel':
        b3, b4 = GBR_interpolated_dict_test, USA_interpolated_dict_test
        b3_val, b4_val = GBR_original_dict_test, USA_original_dict_test
        b7, b8 = GBR_original_dict_test, USA_original_dict_test
    elif args.label_type == 'smooth':
        b3, b4 = GBR_smoothed_dict_test, USA_smoothed_dict_test
        b3_val, b4_val = GBR_smoothed_dict_test, USA_smoothed_dict_test
        b7, b8 = GBR_smoothed_dict_test, USA_smoothed_dict_test
    elif args.label_type == 'smooth_loglabel':
        b3, b4 = GBR_smoothed_dict_test, USA_smoothed_dict_test
        b3_val, b4_val = GBR_interpolated_dict_test, USA_interpolated_dict_test
        b7, b8 = GBR_interpolated_dict_test, USA_interpolated_dict_test
    elif args.label_type == 'smooth_originallabel':
        b3, b4 = GBR_smoothed_dict_test, USA_smoothed_dict_test
        b3_val, b4_val = GBR_original_dict_test, USA_original_dict_test
        b7, b8 = GBR_original_dict_test, USA_original_dict_test
    elif args.label_type == 'raw':
        b3, b4 = GBR_raw_dict, USA_raw_dict
        b3_val, b4_val = GBR_raw_dict, USA_raw_dict
        b7, b8 = GBR_raw_dict, USA_raw_dict
    elif args.label_type == 'original_raw':
        b3, b4 = GBR_original_dict_test, USA_original_dict_test
        b3_val, b4_val = GBR_raw_dict, USA_raw_dict
        b7, b8 = GBR_raw_dict, USA_raw_dict
    elif args.label_type == 'log_raw':
        b3, b4 = GBR_interpolated_dict_test, USA_interpolated_dict_test
        b3_val, b4_val = GBR_raw_dict, USA_raw_dict
        b7, b8 = GBR_raw_dict, USA_raw_dict
    elif args.label_type == 'smooth_raw':
        b3, b4 = GBR_smoothed_dict_test, USA_smoothed_dict_test
        b3_val, b4_val = GBR_raw_dict, USA_raw_dict
        b7, b8 = GBR_raw_dict, USA_raw_dict
    else:
        b3, b4 = GBR_interpolated_dict_test, USA_interpolated_dict_test
        b3_val, b4_val = GBR_interpolated_dict_test, USA_interpolated_dict_test
        b7, b8 = GBR_interpolated_dict_test, USA_interpolated_dict_test

    dict_var = read_concern('related_files/concern-GBR-USA.csv', name=score_version)

    if args.use_feature == 'all':
        position = True
        features = [input_feature, 0, 1, 2, 4,
                    5]  # 0: Abosulute day, 1: totalSeqs, 2: VarSeqs, 3: VarFreq, 4/5: country, 6(True): relevant day
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

    random.seed(1)
    GBR_train_set, GBR_train_label, _, _, _, GBR_linage, GBR_exp_set, exp_linage, GBR_exp_label, GBR_samll_label, GBR_less_label, GBR_lag_label = \
        generate_cross_validation_one_country_new(b1, b3, dict_var, sbar, max_day=max_day, days=day, exp_days=exp_days,
                                                  future=future, ratio=ratio, position=position, features=features,
                                                  input_feature=input_feature, out_feature=out_feature,
                                                  random_split=random_split, token=token,
                                                  prefix_padding=args.prefix_padding, nan_ind=i1,
                                                  smoothed_label=GBR_smoothed_dict_test, noise_level=args.noise_level, drop_rule=args.drop_rule, seq_out=args.seq_out)
    USA_train_set, USA_train_label, _, _, _, USA_linage, USA_exp_set, exp_linage, USA_exp_label, USA_samll_label, USA_less_label, USA_lag_label = \
        generate_cross_validation_one_country_new(b2, b4, dict_var, sbar, max_day=max_day, days=day, exp_days=exp_days,
                                                  future=future, ratio=ratio, position=position, features=features,
                                                  input_feature=input_feature, out_feature=out_feature,
                                                  random_split=random_split, token=token,
                                                  prefix_padding=args.prefix_padding, nan_ind=i2,
                                                  smoothed_label=USA_smoothed_dict_test, noise_level=args.noise_level, drop_rule=args.drop_rule, seq_out=args.seq_out)

    random.seed(1)
    GBR_train_set2, GBR_train_label2, _, GBR_linage2, GBR_exp_set2, exp_linage2, GBR_exp_label2, samll_label2, less_label2, GBR_lag_label2 = \
        generate_cross_validation_one_country_new(b1, b3_val, dict_var, sbar, max_day=max_day2, days=day, exp_days=exp_days,
                                                  future=future, ratio=ratio, position=position, features=features,
                                                  input_feature=input_feature, out_feature=out_feature,
                                                  random_split=random_split, token=token, nan_ind=i1,
                                                  prefix_padding=args.prefix_padding, seq_out=args.seq_out)
    USA_train_set2, USA_train_label2, _, USA_linage2, USA_exp_set2, exp_linage2, USA_exp_label2, samll_label2, less_label2, USA_lag_label2 = \
        generate_cross_validation_one_country_new(b2, b4_val, dict_var, sbar, max_day=max_day2, days=day, exp_days=exp_days,
                                                  future=future, ratio=ratio, position=position, features=features,
                                                  input_feature=input_feature, out_feature=out_feature,
                                                  random_split=random_split, token=token, nan_ind=i2,
                                                  prefix_padding=args.prefix_padding, seq_out=args.seq_out)

    random.seed(1)
    GBR_train_set3, GBR_train_label3, _, GBR_linage3, GBR_exp_set3, exp_linage3, GBR_exp_label3, samll_label3, less_label3, GBR_lag_label3 = \
        generate_cross_validation_one_country_new(b5, b7, dict_var, sbar, max_day=max_day2, days=day, exp_days=exp_days,
                                                  future=future, ratio=1, position=position, features=features,
                                                  input_feature=input_feature, out_feature=out_feature,
                                                  random_split=random_split, token=token, nan_ind=i5,
                                                  prefix_padding=args.prefix_padding, seq_out=args.seq_out)
    USA_train_set3, USA_train_label3, _, USA_linage3, USA_exp_set3, exp_linage3, USA_exp_label3, samll_label3, less_label3, USA_lag_label3 = \
        generate_cross_validation_one_country_new(b6, b8, dict_var, sbar, max_day=max_day2, days=day, exp_days=exp_days,
                                                  future=future, ratio=1, position=position, features=features,
                                                  input_feature=input_feature, out_feature=out_feature,
                                                  random_split=random_split, token=token, nan_ind=i6,
                                                  prefix_padding=args.prefix_padding, seq_out=args.seq_out)

    sat = []

    seed_list = args.seed_list if len(args.seed_list) else list(range(0, 16))  # + [42, 3407]
    pop_list = [(r, r + ratio - 1) for r in range(ratio)] if cv else [(ratio - 1, ratio * 2 - 2)]

    for i, seed in enumerate(seed_list):
        if args.loss_list:
            assert len(args.loss_list) == len(seed_list)
            args.loss = args.loss_list[i]
        torch.manual_seed(seed)

        mae1, mse1, tmae1, tmse1 = 0, 0, 0, 0
        mae2, mse2, tmae2, tmse2 = 0, 0, 0, 0
        
        cmae1, cmae2, cmse1, cmse2 = 0, 0, 0, 0
        tcmae1, tcmae2, tcmse1, tcmse2 = 0, 0, 0, 0

        for p, ind in enumerate(pop_list):

            train_set, train_label = GBR_train_set.copy() + USA_train_set.copy(), GBR_train_label.copy() + USA_train_label.copy()
            train_set.pop(ind[0])
            train_set.pop(ind[1])
            train_label.pop(ind[0])
            train_label.pop(ind[1])

            val_set, val_label = GBR_train_set2.copy() + USA_train_set2.copy(), GBR_train_label2.copy() + USA_train_label2.copy()
            val_name = GBR_linage2.copy() + USA_linage2.copy()
            x_val, y_val, name_val = val_set.pop(ind[0]), val_label.pop(ind[0]), val_name.pop(ind[0])
            x_val2, y_val2, name_val2 = val_set.pop(ind[1]), val_label.pop(ind[1]), val_name.pop(ind[1])
            x_train, y_train = np.concatenate(train_set, axis=0), np.concatenate(train_label, axis=0)
            current_mae1, current_mae2, current_mse1, current_mse2, current_cmae1, current_cmae2, current_cmse1, current_cmse2, model = train(args, seed, x_train, y_train,
                                                                                          x_val, y_val, name_val,
                                                                                          x_val2, y_val2, name_val2,
                                                                                          dims=args.dims, num_p=len(pop_list),
                                                                                          ensemble=False, prints=False)

            mae1 += current_mae1
            mae2 += current_mae2
            mse1 += current_mse1
            mse2 += current_mse2
            cmae1 += current_cmae1
            cmae2 += current_cmae2
            cmse1 += current_cmse1
            cmse2 += current_cmse2

            if args.test_on_new:
                test_set, test_label = np.concatenate(GBR_train_set3, axis=0), np.concatenate(GBR_train_label3, axis=0)
                test_name = GBR_linage3[0]
                _, _, current_tmae1, current_tmse1, current_tcmae1, current_tcmse1, _, _, _ = test(args, model, test_set, test_label, test_name,
                                                                         ensemble=False, prints=False)

                test_set, test_label = np.concatenate(USA_train_set3, axis=0), np.concatenate(USA_train_label3, axis=0)
                test_name = USA_linage3[0]
                _, _, current_tmae2, current_tmse2, current_tcmae2, current_tcmse2, _, _, _ = test(args, model, test_set, test_label, test_name,
                                                                         ensemble=False, prints=False)

                tmae1 += current_tmae1
                tmse1 += current_tmse1
                tmae2 += current_tmae2
                tmse2 += current_tmse2
                
                tcmae1 += current_tcmae1
                tcmse1 += current_tcmse1
                tcmae2 += current_tcmae2
                tcmse2 += current_tcmse2

            if args.save_model:
                to_save = {
                    'model': model.state_dict(),
                    'args': args,
                }
                save_path = f'/{args.ckpt_dir}/{args.used_model+args.ckpt_name}/label_{args.label_type}/seed{seed}/'
                try:
                    os.makedirs(save_path)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
                torch.save(to_save,
                           save_path + f'checkpoint_{args.day}_{args.future}_{args.use_feature}_seed{seed}_{p}.pth')

        print(f'GBR Val: MAE: {mae1 / int(len(pop_list))}, MSE: {mse1 / int(len(pop_list))}')
        print(f'USA Val: MAE: {mae2 / int(len(pop_list))}, MSE: {mse2 / int(len(pop_list))}')
        print(f'GBR Tes: MAE: {tmae1 / int(len(pop_list))}, MSE: {tmse1 / int(len(pop_list))}')
        print(f'USA Tes: MAE: {tmae2 / int(len(pop_list))}, MSE: {tmse2 / int(len(pop_list))}')
        print()

        sat.append([seed, (mse1 + mse2) / int(len(pop_list)), (mae1 + mae2) / int(len(pop_list)),
                    (tmse1 + tmse2) / int(len(pop_list)), (tmae1 + tmae2) / int(len(pop_list))])

        sat.sort(key=lambda x: (x[1]+x[2]))
        print(sat[:10])
        print()
        sat.sort(key=lambda x: (x[3]+x[4]))
        print(sat[:10])
        print()

    dist(sat)

    # GBR_label_list_exp, GBR_pred_list_exp, USA_label_list_exp, USA_pred_list_exp = \
        # exp_test(b1, b2, b3, b4, dict_var, sbar, skip_days,
        #          input_feature, out_feature, step, binary, position, random_split,
        #          future, features, max_day2, exp_days, ratio=5, token=token, prefix_padding=args.prefix_padding)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
