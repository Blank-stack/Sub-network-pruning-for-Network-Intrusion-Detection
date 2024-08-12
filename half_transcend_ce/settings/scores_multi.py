# -*- coding: utf-8 -*-

"""
scores.py
~~~~~~~~~

Functions for producing the various scores used during conformal evaluation,
such as non-conformity measures, credibility and confidence p-values and
probabilities for comparison.

Note that the functions in this module currently only apply to producing
scores for a binary classification task and an SVM classifier. Different
settings and different classifiers will require their own functions for
generating non-conformity measures based on different intuitions.

"""
import multiprocessing

import numpy as np
from tqdm import tqdm


def compute_p_values_cred_and_conf(_probs, y_true, _test_probs, y_test):
    # assert len(set(y_true)) == 2  # binary classification tasks only

    test_cred, test_conf = [], []
    simls_list = []
    # for i in np.unique(y_true):
    for i in range(max(y_true) + 1):
        simls_list.append([])
    # simls_neg = []
    # simls_pos = []
    for t_simls, single_y in zip(_probs, y_true):
        simls_list[single_y].append(t_simls[single_y])
        # if single_y == 0:
        #     simls_neg.append(t_siml_b)
        # else:
        #     simls_pos.append(t_siml_m)

    # nfolds = 10
    # folds = []
    # for index in range(nfolds):
    #     folds.append(index)
    # fold_generator = ({
    #     'simls_neg': simls_neg,
    #     'simls_pos': simls_pos,
    #     'siml0_pack': test_simls_b[int(i * len(y_test) / nfolds):int((i + 1) * len(y_test) / nfolds)][:],
    #     'siml1_pack': test_simls_m[int(i * len(y_test) / nfolds):int((i + 1) * len(y_test) / nfolds)][:],
    #     'y_pack': y_test[int(i * len(y_test) / nfolds):int((i + 1) * len(y_test) / nfolds)][:],
    #     'idx': idx
    # } for idx, i in enumerate(folds))
    #
    # ncpu = multiprocessing.cpu_count()
    # cred_result, conf_result = {}, {}
    # with multiprocessing.Pool(processes=ncpu) as pool:
    #     # n_splits = skf.get_n_splits(test_simls_b, y_test)
    #     for cred_pack, conf_pack, idx in tqdm(pool.imap(pool_compute_cred, fold_generator), total=nfolds):
    #         cred_result[idx] = cred_pack
    #         conf_result[idx] = conf_pack
    # for i in range(nfolds):
    #     test_cred.extend(cred_result[i])
    #     test_conf.extend(conf_result[i])
    #
    # return {'cred': test_cred, 'conf': test_conf}

    # One thread computing p_val
    for simls, y in tqdm(zip(_test_probs, y_test), total=len(y_test),
                         desc='cred_and_conf_s '):
        cred_max, cred_sec = compute_single_cred_set(
            train_simls_list=simls_list,
            # train_simls_neg=simls_neg,
            # train_simls_pos=simls_pos,
            single_test_simls=simls,
            # single_test_siml_b=siml0,
            # single_test_siml_m=siml1,
            single_y=y)
        test_cred.append(cred_max)
        test_conf.append(1 - cred_sec)
    return {'cred': test_cred, 'conf': test_conf}


# def pool_compute_cred(params):
#     simls_neg = params['simls_neg']
#     simls_pos = params['simls_pos']
#     siml0_pack = params['siml0_pack']
#     siml1_pack = params['siml1_pack']
#     y_pack = params['y_pack']
#     idx = params['idx']
#
#     cred_pack = []
#     conf_pack = []
#
#     for siml0, siml1, y in tqdm(zip(siml0_pack, siml1_pack, y_pack), total=len(y_pack),
#                                 desc='cred_and_conf_s {}:'.format(str(idx))):
#         cred_max, cred_sec = compute_single_cred_set(
#             train_simls_neg=simls_neg,
#             train_simls_pos=simls_pos,
#             single_test_siml_b=siml0,
#             single_test_siml_m=siml1,
#             single_y=y)
#         cred_pack.append(cred_max)
#         conf_pack.append(1 - cred_sec)
#     return cred_pack, conf_pack, idx


def compute_single_cred_set(train_simls_list, single_test_simls, single_y):
    # faster
    p_val_list = []
    for i in range(len(train_simls_list)):
        p_val_list.append(compute_single_cred_p_value(train_simls_list[i], single_test_simls[i]))
    cred_max = p_val_list[single_y]
    cred_sec = -10
    for i in range(len(train_simls_list)):
        if i == single_y: continue
        if cred_sec < p_val_list[i]:
            cred_sec = p_val_list[i]

    # # t0 = compute_single_cred_p_value(train_simls_neg, single_test_siml_b)
    # # t1 = compute_single_cred_p_value(train_simls_pos, single_test_siml_m)
    # if single_y == 0:
    #     cred_max = t0
    #     cred_sec = t1
    # else:
    #     cred_max = t1
    #     cred_sec = t0
    return cred_max, cred_sec


def compute_single_cred_p_value(train_simls, single_test_siml):
    if len(train_simls) == 0:
        return 0
    # faster
    how_great_are_the_single_test_siml = len([siml for siml in train_simls if siml < single_test_siml])
    single_cred_p_value = (how_great_are_the_single_test_siml / len(train_simls))
    return single_cred_p_value
