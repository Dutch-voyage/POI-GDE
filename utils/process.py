import json, random
import time as Time
import os
import torch
import pandas as pd
import math
import numpy as np
import pickle as pkl
from matplotlib import pyplot as plt

from tqdm import tqdm


def remap(df: pd.DataFrame, n_user, n_poi):  # 编号
    uid_dict = dict(zip(pd.unique(df['uid']), range(n_user)))
    poi_dict = dict(zip(pd.unique(df['poi']), range(n_poi)))
    cid_dict = dict(zip(pd.unique(df['cid']), range(n_cid)))
    df['uid'] = df['uid'].map(uid_dict)
    df['poi'] = df['poi'].map(poi_dict)
    df['cid'] = df['cid'].map(cid_dict)
    return df

def get_cid_list(df: pd.DataFrame, n_cid, n_loc):
    poi_cid = df.drop_duplicates(subset=['poi'], keep='first', inplace=False).sort_values(
        by=['poi'])[['cid']].values.reshape(-1)
    return poi_cid


if __name__ == "__main__":
    device = torch.device(f'cuda:0')  # 利用GPU做批运算
    random.seed(3407)
    dataSet = "NYC"

    if dataSet == "SG":
        source_pth = 'data/SG_189807_last_100_delUserAndPOILess_10.txt'
        dist_pth = 'data/ode_sg/'
    elif dataSet == "NYC":
        source_pth = 'data/NYC_89385_last_100_delUserAndPOILess_10.txt'
        dist_pth = 'data/ode_nyc/'
    elif dataSet == "TKY":
        source_pth = 'data/TKY_209012_last_100_delUserAndPOILess_10.txt'
        dist_pth = 'data/ode_tky/'

    col_names = ['uid', 'poi', 'cat_id', 'category', 'lat', 'lon', 'offset', 'time', 'unixTime', 'dayOff', 'cid']
    # 只取前4行 ['uid', 'poi', 'lat', 'lon']
    review_df = pd.read_csv(source_pth, sep='\t', header=None, names=col_names,  # .loc[:, ['uid', 'poi', 'lat', 'lon']]
                            encoding='utf8')
    n_user, n_poi, n_cid = pd.unique(review_df['uid']).shape[0], \
                           pd.unique(review_df['poi']).shape[0], pd.unique(review_df['cid']).shape[0]

    review_df = remap(review_df, n_user, n_poi)

    loc_dict = []
    for poi, item in review_df.groupby('poi'):
        lat, lon = item['lat'].iloc[0], item['lon'].iloc[0]
        loc_dict.append([lat, lon])  # poi到经纬度的映射

    print(f'\nData loaded from {source_pth}')

    print(f'User: {n_user}\tPOI: {n_poi}\n')
    print('Start build train set and test set')

    train_set, val_set, test_set, test_val_set = [], [], [], []
    # ===找到home的id类===
    homeCid = None
    for i, rows in review_df.iterrows():
        if rows['category'] == 'Home (private)':
            homeCid = rows['cid']
            break
    prev = []
    latter = []

    def timestamp_datatime(value):
        value = Time.localtime(value)
        dt = [value.tm_year, value.tm_yday, value.tm_mday, value.tm_wday, value.tm_hour + value.tm_min / 60 + value.tm_sec / 3600]
        return dt

    Ysup, Yinf, Msup, Minf, Wsup, Winf, Dsup, Dinf, Hsup, Hinf = 1e7, 0, 1e7, 0, 1e7, 0, 1e7, 0, 1e7, 0
    Latsup, Latinf, Lonsup, Loninf = 1e7, -1e7, 1e7, -1e7

    n_loc = n_poi + n_user
    cid_list = get_cid_list(review_df, n_cid, n_loc)

    for uid, item in tqdm(review_df.groupby('uid')):

        # todo:加上时间差和距离差
        item = item.sort_values(by='unixTime')  # 先按时间排序
        # visitTime = item.loc[:]['unixTime'] % (3600 * 24 * 7)  # by second
        visitTime = item.loc[:]['unixTime']  # by second
        visitTime = visitTime.tolist()
        timestamps = []

        for time in visitTime:
            dt = timestamp_datatime(time)
            timestamps.append(dt)
            Y, M, D, W, H = dt
            Ysup, Yinf = min(Ysup, Y), max(Yinf, Y)
            Msup, Minf = min(Msup, M), max(Minf, M)
            Wsup, Winf = min(Wsup, W), max(Winf, W)
            Dsup, Dinf = min(Dsup, D), max(Dinf, D)
            Hsup, Hinf = min(Hsup, H), max(Hinf, H)

        # temp_geo_traj_dist = np.zeros((len(item)-1, 3))  # [u,v,delta_s]
        # ===求用户中心===
        pos_list = item['poi'].tolist()  # 访问序列[trajLen]
        category_list = item['cid'].tolist()  # 访问poi的类别序列
        category_names = item['category'].tolist()  # 访问poi的类别序列



        test_val_set_i = []
        train_set_i = []
        window_size = 32

        # 8:1:1 train:val:test
        for i in range(window_size, len(pos_list) - 1):  # 为了避免delta为空，从两步开始取样本
            location = (item['lat'].iloc[i], item['lon'].iloc[i])  # 添加一个类别 category_list (lat, lon)
            Latsup, Latinf = min(Latsup, location[0]), max(Latinf, location[0])
            Lonsup, Loninf = min(Lonsup, location[1]), max(Loninf, location[1])
            # if visitTime[i] < visitTime[i - 1]:
            #     visitTime[i] += 3600 * 24 * 7
            if i < 0.8 * (len(pos_list) - window_size - 1) + window_size:
                train_set_i.append((uid, pos_list[i - window_size: i], pos_list[i - window_size + 1: i + 1],
                                        category_list[i - window_size: i], category_list[i - window_size + 1: i + 1],
                                        timestamps[i - window_size: i], timestamps[i - window_size + 1: i + 1],
                                        pos_list[i], category_list[i], timestamps[i], 1))

            if i >= 0.8 * (len(pos_list) - window_size - 1) + window_size:
                test_val_set_i.append((uid, pos_list[i - window_size: i], pos_list[i - window_size + 1: i + 1],
                                       category_list[i - window_size: i], category_list[i - window_size + 1: i + 1],
                                       timestamps[i - window_size: i], timestamps[i - window_size + 1: i + 1],
                                       pos_list[i], category_list[i], timestamps[i], 1))

        train_set.extend(train_set_i)
        test_val_set.extend(test_val_set_i)

    random.shuffle(test_val_set)
    test_set = test_val_set[:int(len(test_val_set) * 0.5)]
    val_set = test_val_set[int(len(test_val_set) * 0.5):]
    # test_set = test_val_set

    batch_size = 1024
    # shuffle


    subjoin_num = batch_size - len(train_set) % batch_size
    subjoin_index = random.choices(range(len(train_set)), k=subjoin_num)
    train_set.extend([train_set[i] for i in subjoin_index])

    subjoin_num = batch_size - len(val_set) % batch_size
    subjoin_index = random.choices(range(len(val_set)), k=subjoin_num)
    val_set.extend([val_set[i] for i in subjoin_index])

    subjoin_num = batch_size - len(test_set) % batch_size
    subjoin_index = random.choices(range(len(test_set)), k=subjoin_num)
    test_set.extend([test_set[i] for i in subjoin_index])

    print(f'size of train set is {len(train_set)}')
    print(f'size of val set is {len(val_set)}')
    print(f'size of test set is {len(test_set)}')

    print(len(loc_dict))

    print(f'Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}')
    print(Ysup, Yinf, Msup, Minf, Wsup, Winf, Dsup, Dinf, Hsup, Hinf)
    print(Latsup, Latinf, Lonsup, Loninf)

    with open(dist_pth + 'processed/param.pkl', 'wb') as f:  #
        pkl.dump(Latsup, f, pkl.HIGHEST_PROTOCOL)
        pkl.dump(Latinf, f, pkl.HIGHEST_PROTOCOL)
        pkl.dump(Lonsup, f, pkl.HIGHEST_PROTOCOL)
        pkl.dump(Loninf, f, pkl.HIGHEST_PROTOCOL)
        pkl.dump(Ysup, f, pkl.HIGHEST_PROTOCOL)
        pkl.dump(Yinf, f, pkl.HIGHEST_PROTOCOL)
        pkl.dump(Msup, f, pkl.HIGHEST_PROTOCOL)
        pkl.dump(Minf, f, pkl.HIGHEST_PROTOCOL)
        pkl.dump(Wsup, f, pkl.HIGHEST_PROTOCOL)
        pkl.dump(Winf, f, pkl.HIGHEST_PROTOCOL)
        pkl.dump(Dsup, f, pkl.HIGHEST_PROTOCOL)
        pkl.dump(Dinf, f, pkl.HIGHEST_PROTOCOL)
        pkl.dump(Hsup, f, pkl.HIGHEST_PROTOCOL)
        pkl.dump(Hinf, f, pkl.HIGHEST_PROTOCOL)
        pkl.dump(loc_dict, f, pkl.HIGHEST_PROTOCOL)
        pkl.dump(n_user, f, pkl.HIGHEST_PROTOCOL)
        pkl.dump(n_poi, f, pkl.HIGHEST_PROTOCOL)
        pkl.dump(n_cid, f, pkl.HIGHEST_PROTOCOL)
        pkl.dump(cid_list, f, pkl.HIGHEST_PROTOCOL)  # poi类别

    with open(dist_pth + f'raw/train.pkl', 'wb') as f:
        pkl.dump(train_set, f, pkl.HIGHEST_PROTOCOL)
    with open(dist_pth + f'raw/test.pkl', 'wb') as f:
        pkl.dump(test_set, f, pkl.HIGHEST_PROTOCOL)
    with open(dist_pth + f'raw/val.pkl', 'wb') as f:
        pkl.dump(val_set, f, pkl.HIGHEST_PROTOCOL)


    print('CTR data dumped\n')
    print('Process done.')

