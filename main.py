import warnings
from my_tools import *
import torch
import copy
# from classifier.MLP import MLP
# from classifier.new_MLP_IDS_plus import MLP_plus
# from classifier.mulilayer_percepton import MyNeutralNet

from half_transcend_ce.half_ce_siml_multi import start_half_transcend
from my_sub_net import train_Solver, mode_switch
from dataloader import load_data

warnings.filterwarnings("ignore")


class my_args:
    def __init__(self, ):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.Train_Num = 50000
        self.Train_LIMIT = 80000  # 训练样本最大数目
        self.Once_process_num = 50000  # 测试窗口-处理
        self.Once_update_num = 10000  # 测试窗口-更新
        self.Overhead_rate = 2 * 0.01

        self.pretrain_iter = 4000
        self.pruning_iter = 2000
        self.retrain_iter = 2000
        self.batch_size = 256
        self.optim = 'Adam'
        self.optim_lr = 1e-3
        # ______以上为通用配置________


def main():
    args = my_args()

    # mode = 'MRM'
    # mode = 'DCWP'
    # mode = 'EVIL'
    # mode = 'EVIL SAM'
    mode = 'ours'
    args = mode_switch(args, mode=mode)

    random_seed = 22
    # random_seed = 55
    # random_seed = 99
    if random_seed is not None:
        set_random_seed(random_seed)

    data_flag = 'Kyoto'
    # data_flag = 'CIC'
    # data_flag = 'CrossNet'
    # _______________________reading dataset____________________
    cache_root = 'G:\_Dataset\##tempXy_cache/'
    X_train, y_train, y_train_distri, full_X, full_y, full_y_distri = load_data(data_flag, cache_root=cache_root)
    # _______________________reading dataset____________________
    if 'CIC' in data_flag:
        args.distri_continual = False
    args.input_dim = len(full_X[0])

    Train_Num = args.Train_Num
    Train_LIMIT = args.Train_LIMIT  # train set max num
    Once_process_num = args.Once_process_num  # window size
    Overhead_rate = args.Overhead_rate  # Oracle overhead

    Processed_packet = Train_Num
    Processed_packet = 0
    # ___________________________________________
    if len(X_train) > Train_Num:
        rand_mask = np.random.choice(np.arange(len(X_train)), Train_Num, replace=False)
        rand_mask = sorted(rand_mask)
        X_train, y_train = X_train[rand_mask], y_train[rand_mask]
        y_train_distri = y_train_distri[rand_mask]
    # ___________________________________________

    pred_list = []
    update_cnt = 0

    # label_uniq_list = ['benign']
    label_uniq_list = []

    def convert_label_to_nums(y_train):
        for i in range(len(y_train)):
            if y_train[i] == 0 or (type(y_train[i]) == type('') and y_train[i].lower() == 'Benign'.lower()):
                y_train[i] = 0
            else:
                k = 1
                while k < len(label_uniq_list):
                    if label_uniq_list[k] == y_train[i]:
                        y_train[i] = k
                        break
                    k += 1

                if k >= len(label_uniq_list):
                    label_uniq_list.append(y_train[i])
                    y_train[i] = k
        return y_train.astype(int)

    if type(y_train[0]) == type(''):
        y_train = convert_label_to_nums(y_train)
        print('y_train is convert to {}'.format(np.unique(y_train)))

    Update_flag = True
    Want_to_update = True
    origin_full_NET, biased_model, debiased_model = None, None, None

    add_X_train, add_y_train, add_y_train_distri = None, None, None
    while Processed_packet < len(full_y):
        X_test = full_X[Processed_packet:Processed_packet + Once_process_num]
        y_test = full_y[Processed_packet:Processed_packet + Once_process_num]
        y_test_distri = full_y_distri[Processed_packet:Processed_packet + Once_process_num]

        if type(y_test[0]) == type(''):
            y_test = convert_label_to_nums(y_test)
            print('y_test is convert to {}'.format(np.unique(y_test)))
        args.output_dim = max(np.concatenate((np.unique(y_train), np.unique(y_test)))) + 1
        args.output_dim_distri = max(np.concatenate((np.unique(y_train_distri), np.unique(y_test_distri)))) + 1
        if args.distri_continual and len(y_train_distri.shape) == 1:
            y_test_distri = y_test_distri.astype(float)
            y_train_distri = y_train_distri.astype(float)
            try:
                add_y_train_distri = add_y_train_distri.astype(float)
            except:
                pass
            args.output_dim_distri = 1

        print('train set size：{}，   test set size：{}'.format(len(X_train), len(X_test)))
        print('output_dim：{}，   output_dim_distri：{}'.format(args.output_dim, args.output_dim_distri))
        print('DATA Generate finished. Starting Now...')

        if Update_flag:
            start_time = now()
            origin_full_NET, biased_model, debiased_model = train_Solver(
                args, X_train, y_train, y_train_distri, add_X_train, add_y_train, add_y_train_distri)

            end_time = now()
            print('Time Used: {}'.format(end_time - start_time))

            total_mem_used = 0  # Bytes
            from deep_master.deeper import deep_getsizeof
            total_mem_used += deep_getsizeof(debiased_model)
            try:
                total_mem_used += deep_getsizeof(biased_model)
            except:
                pass
            print('[***]total_mem_used:   {}  Bytes  =  {}  MB'.format(total_mem_used, total_mem_used / 1024))

            if add_X_train is not None and add_y_train is not None:
                X_train = np.concatenate((X_train, add_X_train))
                y_train = np.concatenate((y_train, add_y_train))
                y_train_distri = np.concatenate((y_train_distri, add_y_train_distri))
        Processed_packet += Once_process_num

        print('Time end at: {}， ---------preded and updated {}---------'.format(end_time, Processed_packet))

        if Processed_packet <= Once_process_num:
            try:
                y_pred = debiased_model.predict(X_train)
            except:
                y_pred = []
                for i in range(100):
                    st = i * len(X_train) // 100
                    ed = (i + 1) * len(X_train) // 100
                    temp_pred = debiased_model.predict(X_train[st:ed])
                    y_pred.extend(temp_pred)
                y_pred = np.array(y_pred)
            evaluate_true_pred_label(y_train, y_pred, desc='training set self-pred',
                                     label_type='plain' if 'CrossNet' in data_flag else '')
        try:
            test_y_pred = debiased_model.predict(X_test)
        except:
            test_y_pred = []
            for i in range(100):
                st = i * len(X_test) // 100
                ed = (i + 1) * len(X_test) // 100
                temp_pred = debiased_model.predict(X_test[st:ed])
                test_y_pred.extend(temp_pred)
            test_y_pred = np.array(test_y_pred)
        evaluate_true_pred_label(y_test, test_y_pred, desc='test prediction',
                                     label_type='plain' if 'CrossNet' in data_flag else '')
        pred_list.extend(test_y_pred)
        if Want_to_update:
            start_time = now()
            # # _________________TRANSCEND__________
            # temp_idxs = np.random.choice(np.arange(len(y_train)), int(len(y_train) * 2 / 3), replace=False)
            # temp_idxs2 = np.zeros(len(y_train)).astype(bool)
            # temp_idxs2[temp_idxs] = True
            #
            # train_X, cal_X = X_train[temp_idxs2], X_train[~temp_idxs2]
            # train_y, cal_y = y_train[temp_idxs2], y_train[~temp_idxs2]
            #
            # train_probs, cal_probs, cal_y_pred, test_probs, test_y_pred = mlp_calculate_ncm(
            #     train_X, cal_X, X_test, debiased_model)
            #
            # # to_csv(train_probs, 'train_probs.csv')
            # # to_csv(train_y, 'train_y_true.csv')
            # # to_csv(cal_probs, 'cal_probs.csv')
            # # to_csv(cal_y, 'cal_y_true.csv')
            # # to_csv(cal_y_pred, 'cal_y_pred.csv')
            # # to_csv(test_probs, 'test_probs.csv')
            # # to_csv(y_test, 'test_y.csv')
            # # to_csv(test_y_pred, 'test_y_predict.csv')
            # mask, reject_rate, order_idx, X_anom_score = start_half_transcend(train_probs, train_y,
            #                                                                   cal_probs, cal_y, cal_y_pred,
            #                                                                   test_probs, y_test, test_y_pred)
            # print('漂移样本比例：{}'.format(reject_rate))
            # # wrong_num = 0
            # Update_idxs = order_idx[:int(Overhead_rate * Once_process_num)]
            # # _________________TRANSCEND__________

            # _________________random__________
            Update_idxs = np.arange(len(X_test))
            np.random.shuffle(Update_idxs)
            Update_idxs = Update_idxs[:int(Overhead_rate * Once_process_num)]
            # _________________random__________

            # # _________________UC__________
            # from UC_selector import probabilistic_uncertainty
            #
            # y_probs = origin_full_NET.predict_proba(X_test)
            # Update_idxs = probabilistic_uncertainty(y_probs, int(Overhead_rate * Once_process_num))
            # # _________________UC__________

            if Update_flag:
                update_cnt += 1
                print('Updated {} times'.format(update_cnt))

                if len(X_train) >= Train_LIMIT:
                    temp_idxs = np.random.choice(np.arange(len(X_train)), Train_LIMIT - len(Update_idxs), replace=False)
                    sorted(temp_idxs)
                    X_train = X_train[temp_idxs]
                    y_train = y_train[temp_idxs]
                    y_train_distri = y_train_distri[temp_idxs]
                add_X_train, add_y_train = X_test[Update_idxs], y_test[Update_idxs]
                add_y_train_distri = y_test_distri[Update_idxs]
                # X_train = np.concatenate((X_train, X_test[Update_idxs]))
                # y_train = np.concatenate((y_train, y_test[Update_idxs]))
            end_time = now()
            print('【*】update Time Used: {}'.format(end_time - start_time))

    print("#########################")
    to_csv(pred_list, './pred_output/pred_list')
    print("#########################")
    print("End.")


if __name__ == '__main__':
    main()
