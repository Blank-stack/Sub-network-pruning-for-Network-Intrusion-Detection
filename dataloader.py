import numpy as np
import pandas as pd
import copy
from tqdm import tqdm
import os
from my_tools import cache_data, load_cached_data, format_path


def load_data(data_flag='Kyoto', cache_root='G:\_Dataset\##tempXy_cache/'):
    if 'Kyoto' in data_flag:
        month_size = 50000
        Kyoto_3_distri_flag = True
        Kyoto_month_flag = False
        Kyoto_iid_paths = [
            'G:/_Dataset/Kyoto2016/Kyoto2016_2006',
            'G:/_Dataset/Kyoto2016/Kyoto2016_2007',
            'G:/_Dataset/Kyoto2016/Kyoto2016_2008',
            'G:/_Dataset/Kyoto2016/Kyoto2016_2009',
            'G:/_Dataset/Kyoto2016/Kyoto2016_2010',
        ]
        Kyoto_near_paths = [
            'G:/_Dataset/Kyoto2016/Kyoto2016_2011',
            'G:/_Dataset/Kyoto2016/Kyoto2016_2012',
            'G:/_Dataset/Kyoto2016/Kyoto2016_2013',
        ]
        Kyoto_far_paths = [
            'G:/_Dataset/Kyoto2016/Kyoto2016_2014',
            'G:/_Dataset/Kyoto2016/Kyoto2016_2015'
        ]
        max_distri_idx = 0
        Kyoto_iid_X, Kyoto_iid_y = None, None
        Kyoto_iid_y_distri = None
        for day_idx, i_path in enumerate(Kyoto_iid_paths):
            temp_X, temp_y = load_pd_csv([i_path], cache_root=cache_root, label_col_at=-2)
            need_idxs = []
            if day_idx == 0:
                month_cnt = 2
            else:
                month_cnt = 12

            for i in range(month_cnt):
                temp_idxs = np.arange(len(temp_X))[i * (len(temp_X) // month_cnt):(i + 1) * (len(temp_X) // month_cnt)]
                temp_idxs = np.random.choice(temp_idxs, month_size, replace=False)
                temp_idxs = sorted(temp_idxs)
                need_idxs.extend(temp_idxs)
                if Kyoto_month_flag:
                    if Kyoto_iid_y_distri is None:
                        Kyoto_iid_y_distri = np.array([max_distri_idx] * len(temp_idxs))
                        max_distri_idx += 1
                    else:
                        Kyoto_iid_y_distri = np.concatenate((Kyoto_iid_y_distri, [max_distri_idx] * len(temp_idxs)))
                        max_distri_idx += 1
            temp_X, temp_y = copy.copy(temp_X[need_idxs]), copy.copy(temp_y[need_idxs])
            if Kyoto_iid_X is None:
                Kyoto_iid_X, Kyoto_iid_y = temp_X, temp_y
                if not Kyoto_month_flag:
                    Kyoto_iid_y_distri = np.array([max_distri_idx] * len(temp_y))
                    max_distri_idx += 1
            else:
                Kyoto_iid_X, Kyoto_iid_y = np.concatenate((Kyoto_iid_X, temp_X)), np.concatenate((Kyoto_iid_y, temp_y))
                if not Kyoto_month_flag:
                    Kyoto_iid_y_distri = np.concatenate((Kyoto_iid_y_distri, [max_distri_idx] * len(temp_y)))
                    max_distri_idx += 1
        if Kyoto_3_distri_flag:
            Kyoto_iid_y_distri = np.zeros(len(Kyoto_iid_y)).astype(int)
        Kyoto_near_X, Kyoto_near_y = None, None
        Kyoto_near_y_distri = None
        for i_path in Kyoto_near_paths:
            temp_X, temp_y = load_pd_csv([i_path], cache_root=cache_root, label_col_at=-2)
            need_idxs = []
            for i in range(12):
                temp_idxs = np.arange(len(temp_X))[i * (len(temp_X) // 12):(i + 1) * (len(temp_X) // 12)]
                temp_idxs = np.random.choice(temp_idxs, month_size, replace=False)
                temp_idxs = sorted(temp_idxs)
                need_idxs.extend(temp_idxs)
                if Kyoto_month_flag:
                    if Kyoto_near_y_distri is None:
                        Kyoto_near_y_distri = np.array([max_distri_idx] * len(temp_idxs))
                        max_distri_idx += 1
                    else:
                        Kyoto_near_y_distri = np.concatenate((Kyoto_near_y_distri, [max_distri_idx] * len(temp_idxs)))
                        max_distri_idx += 1
            temp_X, temp_y = copy.copy(temp_X[need_idxs]), copy.copy(temp_y[need_idxs])
            if Kyoto_near_X is None:
                Kyoto_near_X, Kyoto_near_y = temp_X, temp_y
                if not Kyoto_month_flag:
                    Kyoto_near_y_distri = np.array([max_distri_idx] * len(temp_y))
                    max_distri_idx += 1
            else:
                Kyoto_near_X, Kyoto_near_y = np.concatenate((Kyoto_near_X, temp_X)), np.concatenate(
                    (Kyoto_near_y, temp_y))
                if not Kyoto_month_flag:
                    Kyoto_near_y_distri = np.concatenate((Kyoto_near_y_distri, [max_distri_idx] * len(temp_y)))
                    max_distri_idx += 1
        if Kyoto_3_distri_flag:
            Kyoto_near_y_distri = np.ones(len(Kyoto_near_y)).astype(int)
        Kyoto_far_X, Kyoto_far_y = None, None
        Kyoto_far_y_distri = None
        for i_path in Kyoto_far_paths:
            temp_X, temp_y = load_pd_csv([i_path], cache_root=cache_root, label_col_at=-2)
            need_idxs = []
            for i in range(12):
                temp_idxs = np.arange(len(temp_X))[i * (len(temp_X) // 12):(i + 1) * (len(temp_X) // 12)]
                temp_idxs = np.random.choice(temp_idxs, month_size, replace=False)
                temp_idxs = sorted(temp_idxs)
                need_idxs.extend(temp_idxs)
                if Kyoto_month_flag:
                    if Kyoto_far_y_distri is None:
                        Kyoto_far_y_distri = np.array([max_distri_idx] * len(temp_idxs))
                        max_distri_idx += 1
                    else:
                        Kyoto_far_y_distri = np.concatenate((Kyoto_far_y_distri, [max_distri_idx] * len(temp_idxs)))
                        max_distri_idx += 1
            temp_X, temp_y = copy.copy(temp_X[need_idxs]), copy.copy(temp_y[need_idxs])

            if Kyoto_far_X is None:
                Kyoto_far_X, Kyoto_far_y = temp_X, temp_y
                if not Kyoto_month_flag:
                    Kyoto_far_y_distri = np.array([max_distri_idx] * len(temp_y))
                    max_distri_idx += 1
            else:
                Kyoto_far_X, Kyoto_far_y = np.concatenate((Kyoto_far_X, temp_X)), np.concatenate((Kyoto_far_y, temp_y))
                if not Kyoto_month_flag:
                    Kyoto_far_y_distri = np.concatenate((Kyoto_far_y_distri, [max_distri_idx] * len(temp_y)))
                    max_distri_idx += 1
        if Kyoto_3_distri_flag:
            Kyoto_far_y_distri = 2 * np.ones(len(Kyoto_far_y)).astype(int)
        print(len(Kyoto_iid_X))
        print(len(Kyoto_near_X))
        print(len(Kyoto_far_X))

        X_train, y_train = Kyoto_iid_X, Kyoto_iid_y
        y_train_distri = Kyoto_iid_y_distri
        full_X, full_y = np.concatenate((Kyoto_near_X, Kyoto_far_X)), np.concatenate((Kyoto_near_y, Kyoto_far_y))
        full_y_distri = np.concatenate((Kyoto_near_y_distri, Kyoto_far_y_distri))

    elif 'CIC' in data_flag:
        paths_17 = [
            'G:/_Dataset/CIC_17_day2',
            'G:/_Dataset/CIC_17_day3',
            # 'G:/_Dataset/CIC_17_day4',
            'G:/_Dataset/CIC_17_day5',
        ]
        find_label_list = ['Benign', 'Bot',
                           # 'Web-BruteForce',
                           'DoS GoldenEye', 'DoS Hulk', 'DoS SlowHTTPTest',
                           'DoS Slowloris', 'FTP-BruteForce',
                           # 'Infiltration',
                           # 'Sql Injection',
                           'SSH-BruteForce', ]
        X_train, y_train, y_train_distri = None, None, None
        for i_path in paths_17:
            temp_X, temp_y = load_pd_csv([i_path], cache_root=cache_root)
            # ___________________________________________
            keep_mask = np.zeros(len(temp_y)).astype(bool)
            for idx, i_y in enumerate(temp_y):
                if i_y in find_label_list:
                    keep_mask[idx] = True
            temp_X, temp_y = temp_X[keep_mask], temp_y[keep_mask]
            # ___________________________________________
            if len(np.unique(temp_y)) <= 1: continue
            normal_cnt = np.sum(temp_y == 'Benign')
            abnormal_cnt = np.sum(temp_y != 'Benign')
            print('normal_cnt: {}  |  abnormal_cnt: {}'.format(normal_cnt, abnormal_cnt))
            # ___________________________________________
            if abnormal_cnt * 10 < normal_cnt:
                idxs_A = np.arange(len(temp_X))[temp_y != 'Benign']
                idxs_A = np.random.choice(idxs_A, min(len(idxs_A), 12500), replace=False)
                idxs_B = np.arange(len(temp_X))[temp_y == 'Benign']
                idxs_B = np.random.choice(idxs_B, min(len(idxs_B), 3 * len(idxs_A)), replace=False)
                temp_idxs = sorted(np.concatenate((idxs_A, idxs_B)))
            else:
                temp_idxs = np.random.choice(np.arange(len(temp_X)), 50000, replace=False)
                temp_idxs = sorted(temp_idxs)
            temp_X = temp_X[temp_idxs]
            temp_y = temp_y[temp_idxs]
            normal_cnt = np.sum(temp_y == 'Benign')
            abnormal_cnt = np.sum(temp_y != 'Benign')
            print('normal_cnt: {}  |  abnormal_cnt: {}'.format(normal_cnt, abnormal_cnt))
            for i_label in np.unique(temp_y):
                if i_label == 'Benign': continue
                print('{} cnt:  \t\t{}'.format(i_label, sum(temp_y == i_label)))
            # ___________________________________________
            # ___________________________________________
            print(np.unique(temp_y))
            if X_train is None:
                X_train, y_train = temp_X, temp_y
            else:
                X_train, y_train = np.concatenate((X_train, temp_X)), np.concatenate((y_train, temp_y))
        y_train_distri = np.zeros(len(y_train)).astype(int)

        paths_18 = [
            'G:/_Dataset/CIC_18_1_3',
            'G:/_Dataset/CIC_18_1_4',
            'G:/_Dataset/CIC_18_1_5',
            # # 'G:/_Dataset/CIC_18_2_3',
            # 'G:/_Dataset/CIC_18_2_4',
            # 'G:/_Dataset/CIC_18_2_5',
            # 'G:/_Dataset/CIC_18_3_3',
            # 'G:/_Dataset/CIC_18_3_4',
            'G:/_Dataset/CIC_18_3_5',
        ]
        full_X, full_y, full_y_distri = None, None, None
        for i_path in paths_18:
            temp_X, temp_y = load_pd_csv([i_path], cache_root=cache_root)
            # ___________________________________________
            keep_mask = np.zeros(len(temp_y)).astype(bool)
            for idx, i_y in enumerate(temp_y):
                if i_y in find_label_list:
                    keep_mask[idx] = True
            temp_X, temp_y = temp_X[keep_mask], temp_y[keep_mask]
            # ___________________________________________
            if len(np.unique(temp_y)) <= 1:
                print(i_path)
                continue
            normal_cnt = np.sum(temp_y == 'Benign')
            abnormal_cnt = np.sum(temp_y != 'Benign')
            print('normal_cnt: {}  |  abnormal_cnt: {}'.format(normal_cnt, abnormal_cnt))
            # ___________________________________________
            if abnormal_cnt * 10 < normal_cnt:
                idxs_A = np.arange(len(temp_X))[temp_y != 'Benign']
                idxs_A = np.random.choice(idxs_A, min(len(idxs_A), 10000), replace=False)
                idxs_B = np.arange(len(temp_X))[temp_y == 'Benign']
                idxs_B = np.random.choice(idxs_B, min(len(idxs_B), 4 * len(idxs_A)), replace=False)
                temp_idxs = sorted(np.concatenate((idxs_A, idxs_B)))
            else:
                temp_idxs = np.random.choice(np.arange(len(temp_X)), 50000, replace=False)
                temp_idxs = sorted(temp_idxs)
            temp_X = temp_X[temp_idxs]
            temp_y = temp_y[temp_idxs]
            normal_cnt = np.sum(temp_y == 'Benign')
            abnormal_cnt = np.sum(temp_y != 'Benign')
            print('normal_cnt: {}  |  abnormal_cnt: {}'.format(normal_cnt, abnormal_cnt))
            for i_label in np.unique(temp_y):
                if i_label == 'Benign': continue
                print('{} cnt:  \t\t{}'.format(i_label, sum(temp_y == i_label)))
            # ___________________________________________
            print(np.unique(temp_y))
            if full_X is None:
                full_X, full_y = temp_X, temp_y
            else:
                full_X, full_y = np.concatenate((full_X, temp_X)), np.concatenate((full_y, temp_y))

        full_y_distri = np.ones(len(full_y)).astype(int)

        rand_idxs = np.random.choice(np.arange(len(full_y)), 4 * 50000, replace=False)
        full_X, full_y, full_y_distri = full_X[rand_idxs], full_y[rand_idxs], full_y_distri[rand_idxs]

    elif 'CrossNet' in data_flag:
        paths = [
            'G:/_Dataset/CrossNet2021_ProGraph/static feature2/ScenarioA',
            'G:/_Dataset/CrossNet2021_ProGraph/static feature2/ScenarioB',
        ]
        X_train, y_train, y_train_distri = None, None, None
        for i_path in [paths[0]]:
            temp_X, temp_y = load_pd_csv([i_path], cache_root=cache_root, label_col_at=-2)
            if X_train is None:
                X_train, y_train = temp_X, temp_y
            else:
                X_train, y_train = np.concatenate((X_train, temp_X)), np.concatenate((y_train, temp_y))
        y_train_distri = np.zeros(len(y_train)).astype(int)

        for i_label in np.unique(temp_y):
            if i_label == 'Benign': continue
            print('{} cnt:  \t\t{}'.format(i_label, sum(temp_y == i_label)))

        full_X, full_y = None, None
        for i_path in [paths[1]]:
            temp_X, temp_y = load_pd_csv([i_path], cache_root=cache_root, label_col_at=-2)
            if full_X is None:
                full_X, full_y = temp_X, temp_y
            else:
                full_X, full_y = np.concatenate((full_X, temp_X)), np.concatenate((full_y, temp_y))
        full_y_distri = np.ones(len(full_y)).astype(int)

        for i_label in np.unique(temp_y):
            if i_label == 'Benign': continue
            print('{} cnt:  \t\t{}'.format(i_label, sum(temp_y == i_label)))

        rand_idxs = np.random.choice(np.arange(len(full_y)), len(full_y), replace=False)
        full_X, full_y, full_y_distri = full_X[rand_idxs], full_y[rand_idxs], full_y_distri[rand_idxs]
    else:
        print('Not knowing what dataset to use...')
        exit(-1)
    return X_train, y_train, y_train_distri, full_X, full_y, full_y_distri


def load_pd_csv(paths=['G:/_Dataset/CIC_17_day2'],
                label_col_at=-1,
                formats='.csv',
                cache_root='G:\_Dataset\##tempXy_cache/'):
    print(f'Loading dataset by pandas...')

    full_X, full_y = None, None
    for i_path in paths:
        if os.path.exists(cache_root + i_path.split('/')[-1] + '.p'):
            data = load_cached_data(cache_root + i_path.split('/')[-1] + '.p')
            tempX, tempy = data['tempX'], data['tempy']
            print('total len:{}, feature_size = {}'.format(len(tempy), len(tempX[0])))
        else:
            path = format_path(i_path, formats)
            source = pd.read_csv(path, index_col=None)
            width = source.shape[1]
            label_col_at = label_col_at if label_col_at < 0 else label_col_at - width

            tempX, tempy = source.iloc[:, :width + label_col_at].values, source.iloc[:, width + label_col_at].values
            print('total len:{}, feature_size = {}'.format(len(tempy), len(tempX[0])))

            data = {'tempX': tempX, 'tempy': tempy}
            cache_data(data, cache_root + i_path.split('/')[-1] + '.p')

        if full_X is None and full_y is None:
            full_X = copy.copy(tempX)
            full_y = copy.copy(tempy)
        else:
            full_X = np.concatenate((full_X, copy.copy(tempX)))
            full_y = np.concatenate((full_y, copy.copy(tempy)))
    print(f'Finnished load dataset by pandas.')
    return full_X, full_y
