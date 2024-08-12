import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import copy
from classifier.new_MLP_IDS_plus import MLP_plus, WeightedCrossEntropyLoss
from tqdm import tqdm


def train_MLP(args, X_train, y_train):
    input_dim = X_train.shape[1]
    hidden_dim = [100, 100, 32]
    output_dim = args.output_dim
    output_dim_distri = args.output_dim_distri

    origin_full_NET = MLP_plus(input_dim, hidden_dim, output_dim, output_dim_distri,
                               epoch=args.pretrain_iter, optim_lr=args.optim_lr, init_mask=args.init_mask,
                               distri_continual=args.distri_continual).to(args.device)
    not_training_Net = copy.deepcopy(origin_full_NET)
    origin_full_NET.fit(X_train, y_train, weights=None)
    return origin_full_NET.to(args.device), not_training_Net


def get_weight_array(args, input_X, input_y, input_y_distri):
    input_dim = input_X.shape[1]
    hidden_dim = [100, 100, 32]
    output_dim = args.output_dim
    output_dim_distri = args.output_dim_distri

    upweight_array = np.ones(len(input_X)).astype(int)
    if 'bias'.lower() in args.wrong_mask_flag.lower():
        bias_NET = MLP_plus(input_dim, hidden_dim, output_dim, output_dim_distri, loss_func=args.wrong_mask_flag,
                            epoch=args.pretrain_iter, optim_lr=args.optim_lr, init_mask=0,
                            distri_continual=args.distri_continual).to(args.device)
        bias_NET.fit(input_X, input_y, weights=None)

        bias_pred = bias_NET.predict(input_X)
        bias_pred_wrong_mask = bias_pred != input_y

        # weights[wrong_label_indices] = 80
        upweight_array[bias_pred_wrong_mask] = len(input_X) / sum(bias_pred_wrong_mask)
    else:
        print('Not setting correct wrong_mask type')
    return upweight_array


def train_PRUNE(args, A_Net, features, labels, distri_labels, upweight_array=None):
    A_Net.train()
    weights = torch.ones(len(labels), dtype=torch.float32).to(args.device)
    if upweight_array is not None:
        weights = torch.tensor(upweight_array, dtype=torch.float32).to(args.device)

    optim_lr_step = 5
    optim_lr_num = 1
    optim_lr = args.optim_lr * pow(optim_lr_step, optim_lr_num - 1)
    optimizer = optim.Adam(A_Net.parameters(), lr=optim_lr)
    A_Net.to(args.device)
    A_Net.pruning_switch(True)

    features = torch.tensor(features, dtype=torch.float32).to(args.device)
    labels = torch.tensor(labels, dtype=torch.float32).to(args.device)
    distri_labels = torch.tensor(distri_labels, dtype=torch.float32).to(args.device)

    for epoch in tqdm(range(args.pruning_iter)):
        if (epoch + 1) % (args.pruning_iter // optim_lr_num) == 0:
            optim_lr /= optim_lr_step
            optimizer = optim.Adam(A_Net.parameters(), lr=optim_lr)

        batch_index = A_Net.next_batches()
        xs, ws = features[batch_index], weights[batch_index]
        y1s, y2s = labels[batch_index], distri_labels[batch_index]

        criterion = WeightedCrossEntropyLoss()
        outputs, last_out = A_Net.forward(xs, need_feature=True)
        loss_main_1 = criterion(outputs.view(-1, A_Net.output_dim), y1s, ws)
        outputs, last_out = A_Net.forward(xs, need_feature=True, need_flip=True)

        if not args.distri_flag:
            loss_main_2 = torch.tensor(1, dtype=torch.float32).to(args.device)
        else:
            if not args.distri_continual:
                # Spatial
                loss_main_2 = criterion(outputs.view(-1, A_Net.output_dim_distri), y2s, ws)
            else:
                # Temporal
                delta = (outputs.view(-1, ) - y2s.long().to(args.device))
                loss_main_2 = delta * delta * ws
                loss_main_2 = loss_main_2.mean()

        if epoch == 0:
            cali_1 = 1 / (loss_main_1.item())
            cali_2 = 1 / (loss_main_2.item())
        loss = cali_1 * loss_main_1 + args.lambda_prune * cali_2 * loss_main_2
        if epoch % (args.pruning_iter // 5) == 0:
            print('Epoch [{}/{}] | loss_main_1: {} | loss_main_2: {}'.format(
                epoch + 1, args.pruning_iter, cali_1 * loss_main_1.item(), cali_2 * loss_main_2.item()
            ))
            # Count parameters before reinitialization
            total_params_before, effective_params_before = A_Net.count_parameters()
            print(f"Effective parameters: {effective_params_before}/{total_params_before}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    A_Net.pruning_switch(False)


def retrain(args, A_Net, features, labels, distri_labels, freeze=True):
    A_Net.train()
    weights = torch.ones(len(labels), dtype=torch.float32).to(args.device)

    optimizer = optim.Adam(A_Net.parameters(), lr=A_Net.optim_lr)
    A_Net.to(args.device)
    A_Net.pruning_switch(False)
    A_Net.freeze_switch(freeze)

    features = torch.tensor(features, dtype=torch.float32).to(args.device)
    labels = torch.tensor(labels, dtype=torch.float32).to(args.device)

    for epoch in tqdm(range(args.retrain_iter)):
        batch_index = A_Net.next_batches()
        xs, ys, ws = features[batch_index], labels[batch_index], weights[batch_index]

        outputs, last_out = A_Net.forward(xs, need_feature=True)

        criterion = WeightedCrossEntropyLoss()
        loss_main = criterion(outputs.view(-1, A_Net.output_dim), ys, ws)

        if args.con_retrain.lower() == 'SupCon'.lower():
            loss_con = A_Net.con_criterion(F.normalize(last_out, dim=1).unsqueeze(1), ys)
        elif args.con_retrain.lower() == 'SAM'.lower():
            l2_reg = 0.001
            weight_penalty_params = A_Net.parameters()
            # weight_penalty_params = list(weight_penalty_params)
            weight_l2 = sum([torch.sum(x ** 2) for x in weight_penalty_params])
            loss_con = l2_reg * 0.5 * weight_l2
        else:
            loss_con = torch.tensor(1)

        if epoch == 0:
            cali_main = 1 / (loss_main.item())
            cali_con = 1 / (loss_con.item())
        loss = cali_main * loss_main + args.lambda_retrain * cali_con * loss_con
        if epoch % (args.retrain_iter // 5) == 0:
            print('Epoch [{}/{}] | loss_main: {} | loss_con: {}'.format(
                epoch + 1, args.pruning_iter, cali_main * loss_main.item(), cali_con * loss_con.item()
            ))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    A_Net.pruning_switch(False)
    A_Net.freeze_switch(freeze)


def train_Solver(args, X_train, y_train, y_dis_train,
                 add_X_train=None, add_y_train=None, add_y_dis_train=None):
    if add_X_train is not None and add_y_train is not None:
        full_X_train = np.concatenate((X_train, add_X_train))
        full_y_train = np.concatenate((y_train, add_y_train))
        full_y_dis_train = np.concatenate((y_dis_train, add_y_dis_train))
    print("#########  Pre-Training  ##########")
    if add_X_train is not None and add_y_train is not None:
        debias_NET, not_training_Net = train_MLP(args, full_X_train, full_y_train)
    else:
        debias_NET, not_training_Net = train_MLP(args, X_train, y_train)
    print('---------  End with Pre-Training  -------')
    # origin_full_NET = copy.deepcopy(debias_NET)
    origin_full_NET = not_training_Net

    # pred_test_pre_debias = self.predict(X_test)
    # evaluate_true_pred_label(y_test, pred_test_pre_debias, desc='Pre-Training')
    ###################  End with Pre-Training  ###################################

    print("#########  Pruning  ##########")
    if args.wrong_mask_flag is not None:
        if add_X_train is not None and add_y_train is not None:
            upweight_array = get_weight_array(args, full_X_train, full_y_train, full_y_dis_train)
        else:
            upweight_array = get_weight_array(args, X_train, y_train, y_dis_train)
    else:
        upweight_array = None

    if add_X_train is not None and add_y_train is not None:
        train_PRUNE(args, debias_NET, full_X_train, full_y_train, full_y_dis_train, upweight_array)
    else:
        train_PRUNE(args, debias_NET, X_train, y_train, y_dis_train, upweight_array)
    print('---------  End with Pruning  -------')

    # pred_test_PRUNE_debias = self.predict(X_test)
    # evaluate_true_pred_label(y_test, pred_test_PRUNE_debias, desc='Pruning')
    ###################  End with Pruning  ###################################

    # Count parameters before reinitialization
    total_params_before, effective_params_before = debias_NET.count_parameters()
    print(f"Effective parameters: {effective_params_before}/{total_params_before}")
    # ################### MRM adjustment  #####################################
    # reinitialize the weight, overlay, and retrain
    if args.reinitialize:
        for (name1, a_GateMLP1), (name2, a_GateMLP2) in zip(
                origin_full_NET.named_modules(), debias_NET.named_modules()):
            from prune.GateLayer import GateMLP
            if not isinstance(a_GateMLP1, GateMLP):
                continue
            a_GateMLP2.weight.data = a_GateMLP1.weight.data

        print('Reinitialized model.')
    # ###################  End with MRM adjustment #####################################

    if args.retrain_flag:
        print("#########  Retraining  ##########")
        if add_X_train is not None and add_y_train is not None:
            retrain(args, debias_NET, full_X_train, full_y_train, full_y_dis_train,
                    freeze=True and not args.reinitialize)
        else:
            retrain(args, debias_NET, X_train, y_train, y_dis_train, freeze=True and not args.reinitialize)
        print('---------  End with Retraining  -------')

    # pred_test_retrain_debias = self.predict(X_test)
    # evaluate_true_pred_label(y_test, pred_test_retrain_debias, desc='Retraining')
    ###################  End with Retraining  ###################################
    return origin_full_NET, None, debias_NET


def mode_switch(args, mode='ours'):
    # args.distri_weight_add_flag = False
    args.wrong_mask_flag = None
    args.init_mask = 0.
    args.lambda_prune = 1
    args.lambda_retrain = 1
    if 'MRM' in mode:
        args.distri_flag = False
        args.distri_continual = False
        args.reinitialize = True
        args.retrain_flag = True
        args.con_retrain = 'None'
    elif 'DCWP' in mode:
        args.wrong_mask_flag = 'bias'
        args.distri_flag = False
        args.distri_continual = False
        args.reinitialize = False
        args.retrain_flag = True
        args.con_retrain = 'SupCon'
    elif 'EVIL' in mode:
        args.distri_flag = True
        args.distri_continual = False
        args.reinitialize = False
        args.retrain_flag = True
        if 'SAM' in mode:
            args.con_retrain = 'SAM'
        else:
            args.con_retrain = 'EVIL'
    elif 'ours' in mode:
        # args.wrong_mask_flag = 'gradual_bias'
        args.wrong_mask_flag = 'bias'
        # # args.distri_weight_add_flag = True
        # args.distri_weight_add_flag = False
        args.distri_flag = True
        args.distri_continual = True
        args.reinitialize = False
        args.retrain_flag = True
        args.con_retrain = 'SupCon'

    if not args.distri_flag:  args.distri_continual = False

    if 'EVIL' in mode and 'SAM' in mode:
        args.lambda_retrain = 1

    # # ______________________________________
    # if 'gradual_bias' in args.wrong_mask_flag:
    #     args.wrong_mask_flag += ' 0.2'
    #     # args.wrong_mask_flag += ' 0.00'
    #     # args.wrong_mask_flag += ' 0.05'
    #     # args.wrong_mask_flag += ' 0.1'
    #     # args.wrong_mask_flag += ' 0.2'
    #     # args.wrong_mask_flag += ' 0.5'
    #     # args.wrong_mask_flag += ' 1.0'

    # ______________________________________
    # if not args.distri_flag: args.lambda_prune = None
    # if not args.con_retrain: args.lambda_prune = None
    # ______________________________________
    if 'ours' in mode:
        args.lambda_prune = 10
        # args.lambda_prune = 0.01
        # args.lambda_prune = 0.1
        # args.lambda_prune = 1
        # args.lambda_prune = 10
        # args.lambda_prune = 100
        # args.lambda_prune = 1000
        # # args.lambda_prune = 0.001
        # ______________________________________
        args.lambda_retrain = 1
        # args.lambda_retrain = 0.001
        # args.lambda_retrain = 0.01
        # args.lambda_retrain = 0.1
        # args.lambda_retrain = 1
        # args.lambda_retrain = 10
        # args.lambda_retrain = 100
        # # args.lambda_retrain = 0.0001
        # args.lambda_prune = 0
        # args.lambda_retrain = 0
        args.distri_continual = False

    # # # args.reinitialize = False
    # # args.reinitialize = True
    # args.wrong_mask_flag = None
    if 'ours' in mode:
        args.init_mask = 0.
        # args.init_mask = 0.1
        # args.init_mask = 0.05
        # args.init_mask = 0.
        # args.init_mask = -0.05
    return args
