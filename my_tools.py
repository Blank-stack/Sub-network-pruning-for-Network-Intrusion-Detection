import os, pickle, csv
import numpy as np
from timeit import default_timer as timer
from sklearn import metrics as metrics
import shutil
from tqdm import tqdm


def now():
    from datetime import datetime
    return datetime.now()


def del_dir(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)


def format_path(data_path, Itype='.p'):
    '''Used for adding "Itype" '''
    if '.' not in data_path or data_path.split('.')[-1] != Itype[1:]:
        print('[*] adding {}'.format(Itype))
        data_path += Itype
    folder_path = os.path.dirname(data_path)
    if folder_path != '' and not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return data_path


def cache_data(data, data_path):
    data_path = format_path(data_path, Itype='.p')

    with open(data_path, 'wb') as f:
        pickle.dump(data, f)
    print('Done cache_data for {}.'.format(data_path))


def load_cached_data(data_path):
    data_path = format_path(data_path, Itype='.p')

    with open(data_path, 'rb') as f:
        model = pickle.load(f)
    print('Done load_cached_data for {}.'.format(data_path))
    return model


def set_random_seed(seed=42, deterministic=True):
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except:
        pass


def to_csv(pred, csv_path):
    csv_path = format_path(csv_path, Itype='.csv')
    if (type(pred[0]) != type([])) and (type(pred[0]) != type(np.array([]))):
        pred = np.array(pred).reshape(-1, 1)
    # pred = np.array(pred).reshape(-1, col_num)

    with open(csv_path, 'w', newline='') as f:
        f_csv_writer = csv.writer(f)

        f_csv_writer.writerows(pred)
        f.close()


def evaluate_true_pred_label(y_true, y_pred, desc='', para='strong', label_type=''):
    num = y_true.shape[0]
    if num == 0:
        return None
    print('-' * 10 + desc + '-' * 10)
    if 'plain' in label_type:
        from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
        acc = accuracy_score(y_true, y_pred)
        pre = recall_score(y_true, y_pred, average="weighted")
        rec = precision_score(y_true, y_pred, average="weighted")
        f1 = f1_score(y_true, y_pred, average="weighted")

        print("\tRecall \t{:6.4f}".format(rec), end='\t - ')
        print("\tPrecision \t{:6.4f}".format(pre))
        print("\tAccuracy \t{:6.4f}".format(acc), end='\t - ')
        print("\tF1 \t{:6.4f}".format(f1))

    else:
        y_true = np.array([0 if xxx == 0 else 1 for xxx in y_true])
        y_pred = np.array([0 if xxx == 0 else 1 for xxx in y_pred])

        cf_flow = metrics.confusion_matrix(y_true, y_pred)
        if len(cf_flow.ravel()) == 1:
            if y_true[0] == 0:
                TN, FP, FN, TP = cf_flow[0][0], 0, 0, 0
            elif y_true[0] == 1:
                TN, FP, FN, TP = 0, 0, 0, cf_flow[0][0]
            else:
                raise Exception("label error")
        else:
            TN, FP, FN, TP = cf_flow.ravel()

        rec = (TP / (TP + FN)) if (TP + FN) != 0 else 0
        prec = (TP / (TP + FP)) if (TP + FP) != 0 else 0
        Accu = (TP + TN) / len(y_true)

        F1 = 2 * rec * prec / (rec + prec) if (rec + prec) != 0 else 0
        if para.lower() == 'strong'.lower():
            print("TP:\t" + str(TP), end='\t|| ')
            print("FP:\t" + str(FP), end='\t|| ')
            print("TN:\t" + str(TN), end='\t|| ')
            print("FN:\t" + str(FN))
            print("Recall:\t{:6.4f}".format(rec), end='\t|| ')
            print("Precision:\t{:6.4f}".format(prec))
            print("Accuracy:\t{:6.4f}".format(Accu), end='\t|| ')
            print("F1:\t{:6.4f}".format(F1))
        else:
            print("\tTP \t" + str(TP), end='\t - ')
            print("\tFP \t" + str(FP), end='\t - ')
            print("\tTN \t" + str(TN), end='\t - ')
            print("\tFN \t" + str(FN))
            print("\tRecall \t{:6.4f}".format(rec), end='\t - ')
            print("\tPrecision \t{:6.4f}".format(prec))
            print("\tAccuracy \t{:6.4f}".format(Accu), end='\t - ')
            print("\tF1 \t{:6.4f}".format(F1))


from sys import getsizeof, stderr
from itertools import chain
from collections import deque


def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.
    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:
        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}
    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                    }
    all_handlers.update(handlers)  # user handlers take precedence
    seen = set()  # track which object id's have already been seen
    default_size = getsizeof(0)  # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:  # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)


def mlp_calculate_ncm(train_X, cal_X, X_test, model):
    # import torch
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model.to('cpu')
    train_prob = model.predict_proba(train_X)
    # train_b = train_prob[:, 0]
    # train_m = train_prob[:, 1]
    cal_prob = model.predict_proba(cal_X)
    # cal_b = cal_prob[:, 0]
    # cal_m = cal_prob[:, 1]
    cal_y_pred = model.predict(cal_X)

    test_prob = model.predict_proba(X_test)
    # test_b = test_prob[:, 0]
    # test_m = test_prob[:, 1]
    test_y_pred = model.predict(X_test)

    return train_prob, cal_prob, cal_y_pred, test_prob, test_y_pred


if __name__ == '__main__':
    pass
    y_true = [3, 0]
    y_pred = [2, 3]
    evaluate_true_pred_label(y_true, y_pred, desc='')
