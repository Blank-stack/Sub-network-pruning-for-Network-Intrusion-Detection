# -*- coding: utf-8 -*-

"""
thresholding.py
~~~~~~~~~~~~~~~

Functions for deriving and applying thresholds during conformal evaluation.

"""
import logging
import statistics

import numpy as np
from sklearn import metrics as metrics
from termcolor import cprint
from tqdm import tqdm
from itertools import repeat

import multiprocessing as mp
import os


def test_with_rejection(
        multi_thresholds, test_scores, groundtruth_labels, predicted_labels, full=True):
    """Get test results, rejecting predictions based on a given threshold.

    `binary_thresholds` expects a dictionary keyed by 'cred' and/or 'conf',
    with sub-dictionaries containing the thresholds for the mw and gw classes.

    Note that the keys of `binary_thresholds` determine _which_ thresholding
    criteria will be enforced. That is, if only a 'cred' dictionary is supplied
    thresholding will be enforced on cred-only and the same for 'conf'.
    Supplying cred and conf dictionaries will enforce the 'cred+conf'
    thresholding criteria (all thresholds will be applied).

    `test_scores` expects a dictionary in much the same way, with at least the
    same keys as `binary_thresholds` ('cred' and/or 'conf' at the top level).

    See Also:
        - `apply_threshold`
        - `get_performance_with_rejection`

    Args:
        binary_thresholds (dict): The threshold to apply.
        test_scores (dict): The test scores to apply the threshold to.
        groundtruth_labels (np.ndarray): The groundtruth label for each object.
        predicted_labels (np.ndarray): The set of predictions to decide which
            'per-class' threshold to use. Depending on the stage of conformal
            evaluation, this could be either the predicted or ground truth
            labels.
        full (boolean): Optimization flag which dictates how much data to return,
            default is True. False gives a lot more performance but removes a lot 
            of metrics. 

    Returns:
        dict: A dictionary of results for baseline, kept, and rejected metrics.

    """
    keep_mask = apply_threshold(
        multi_thresholds=multi_thresholds,
        test_scores=test_scores,
        y_test=predicted_labels)

    results = get_performance_with_rejection(
        y_true=groundtruth_labels,
        y_pred=predicted_labels,
        keep_mask=keep_mask,
        full=full)

    return results, keep_mask, predicted_labels


def apply_threshold(multi_thresholds, test_scores, y_test):
    """Returns a 'keep mask' describing which elements to include.

    Elements that fall above the threshold (and should be kept) have
    their indexes marked TRUE.

    Elements that fall below the threshold (and should be rejected) have
    their indexes marked FALSE.

    `binary_thresholds` expects a dictionary keyed by 'cred' and/or 'conf',
    with sub-dictionaries containing the thresholds for the mw and gw classes.

    Note that the keys of `binary_thresholds` determine _which_ thresholding
    criteria will be enforced. That is, if only a 'cred' dictionary is supplied
    thresholding will be enforced on cred-only and the same for 'conf'.
    Supplying cred and conf dictionaries will enforce the 'cred+conf'
    thresholding criteria (all thresholds will be applied).

    `test_scores` expects a dictionary in much the same way, with at least the
    same keys as `binary_thresholds` ('cred' and/or 'conf' at the top level).

    Example:
        >>> thresholds = {'cred': {'mw': 0.4, 'gw': 0.6},
        ...               'conf': {'mw': 0.5, 'gw': 0.8}}
        >>> scores = {'cred': [0.4, 0.2, 0.7, 0.8, 0.6],
        ...           'conf': [0.6, 0.8, 0.3, 0.2, 0.4]}
        >>> y = np.array([1, 1, 1, 0, 0])
        >>> apply_threshold(thresholds, scores, y)
        array([ True, False, False, False, False])

    Args:
        binary_thresholds(dict): The threshold to apply.
        test_scores (dict): The test scores to apply the threshold to.
        y_test (np.ndarray): The set of predictions to decide which 'per-class'
            threshold to use. Depending on the stage of conformal evaluation,
            this could be either the predicted or ground truth labels.

    Returns:
        np.ndarray: Boolean mask to use on the elements (1 = kept, 0 = reject).

    """
    # Assert preconditions
    assert (set(multi_thresholds.keys()) in
            [{'cred'}, {'conf'}, {'cred', 'conf'}])

    for key in multi_thresholds.keys():
        assert key in test_scores.keys()
        # assert set(binary_thresholds[key].keys()) == {'mw', 'gw'}

    def get_class_threshold(criteria, k):
        return multi_thresholds[criteria][f'class{k}']
        # return (binary_thresholds[criteria]['mw'] if k == 1
        #         else binary_thresholds[criteria]['gw'])

    keep_mask = []
    for i, y_prediction in enumerate(y_test):

        cred_threshold, conf_threshold = 0, 0
        current_cred, current_conf = 0, 0

        if 'cred' in multi_thresholds:
            key = 'cred'
            current_cred = test_scores[key][i]
            cred_threshold = get_class_threshold(key, y_prediction)

        if 'conf' in multi_thresholds:
            key = 'conf'
            current_conf = test_scores[key][i]
            conf_threshold = get_class_threshold(key, y_prediction)

        keep_mask.append(
            (current_cred >= cred_threshold) and
            (current_conf >= conf_threshold))

    return np.array(keep_mask, dtype=bool)


def find_quartile_thresholds(
        scores, predicted_labels, groundtruth_labels, consider='correct'):
    """Find the quartile thresholds for a given set of scores.

    Here we find thresholds on the correct predictions only and  return
    the per-class thresholds (mw/gw) for q1, q2, q3, and mean.

    Args:
        scores (np.ndarray): The set of scores on which to apply the threshold.
        predicted_labels (np.ndarray): The prediction outcome for each object.
        groundtruth_labels (np.ndarray): The groundtruth label for each object.
        consider (str): ['correct'|'incorrect'|'all']. Whether to consider only
            correct predictions, incorrect predictions, or not to distinguish
            between them.

    Returns:
        dict: Set of thresholds for quartiles and mean.
            Keyed by points: 'q1', 'q2', 'q3', 'mean' at the first level and
            'mw', 'gw' at the second level.

    """
    scores_list = sort_by_predicted_label(
        scores, predicted_labels, groundtruth_labels, consider=consider)
    temp_dict = {}
    # for i in range(len(np.unique(groundtruth_labels))):
    for i in range(max(groundtruth_labels) + 1):
        temp_dict[f'class{i}'] = np.percentile(scores_list[i], 5) if len(scores_list[i]) > 0 else 0
    thresholds = {
        # 'q1': {
        #     'mw': np.percentile(scores_mw, 25),
        #     'gw': np.percentile(scores_gw, 25)
        # },
        # 'q2': {
        #     'mw': np.percentile(scores_mw, 50),
        #     'gw': np.percentile(scores_gw, 50)
        # },
        # 'q3': {
        #     'mw': np.percentile(scores_mw, 75),
        #     'gw': np.percentile(scores_gw, 75)
        # },
        # 'mean': {
        #     'mw': np.mean(scores_mw),
        #     'gw': np.mean(scores_gw)
        # },
        # 'aus1': {
        #     'mw': np.percentile(scores_mw, 5) if len(scores_mw) > 0 else 0,
        #     'gw': np.percentile(scores_gw, 5) if len(scores_gw) > 0 else 0
        # }
        'aus1': temp_dict
        # 'aus2': {
        #     'mw': np.percentile(scores_mw, 5),
        #     'gw': np.percentile(scores_gw, 10)
        # },
        # 'aus3': {
        #     'mw': np.percentile(scores_mw, 1),
        #     'gw': np.percentile(scores_gw, 10)
        # }
    }
    return thresholds


def sort_by_predicted_label(
        scores, predicted_labels, groundtruth_labels, consider='correct'):
    """Sort scores into lists of their respected predicted classes.

    Divide a set of scores into 'predicted positive' and 'predicted
    negative' results. Optionally consider only correct or incorrect
    predictions. `scores`, `predicted_labels`, and `groundtruth_labels`
    should be aligned (one per observation).

    Example:
        >>> s = np.array([0.8, 0.7, 0.6, 0.9])
        >>> y_pred = np.array([1, 1, 0, 0])
        >>> y_true = np.array([1, 0, 1, 0])
        >>> sort_by_predicted_label(s, y_pred, y_true, 'correct')
        (array([0.8]), array([0.9]))
        >>> sort_by_predicted_label(s, y_pred, y_true, 'incorrect')
        (array([0.7]), array([0.6]))
        >>> sort_by_predicted_label(s, y_pred, y_true, 'all')
        (array([0.8, 0.7]), array([0.6, 0.9]))

    Args:
        scores (np.ndarray): Predicted scores to be sorted.
        predicted_labels (np.ndarray): The prediction outcome for each object.
        groundtruth_labels (np.ndarray): The groundtruth label for each object.
        consider (str): ['correct'|'incorrect'|'all']. Whether to consider only
            correct predictions, incorrect predictions, or not to distinguish
            between them.

    Returns:
        (np.ndarray, np.ndarray): Tuple of sorted scores (malware, goodware).

    """

    def predicted(i, k):
        return predicted_labels[i] == k

    def correct(i, k):
        return predicted(i, k) and (groundtruth_labels[i] == k)

    def incorrect(i, k):
        return predicted(i, k) and (groundtruth_labels[i] == (k ^ 1))

    if consider == 'all':
        select = predicted
    elif consider == 'correct':
        select = correct
    elif consider == 'incorrect':
        select = incorrect
    else:
        raise ValueError('Unknown thresholding criteria!')

    scores_list = []
    # for kkk in range(len(np.unique(groundtruth_labels))):
    for kkk in range(max(groundtruth_labels) + 1):
        scores_list.append(np.array([scores[i] for i in range(len(scores)) if select(i, kkk)]))
    return scores_list
    # scores_mw = [scores[i] for i in range(len(scores)) if select(i, 1)]
    # scores_gw = [scores[i] for i in range(len(scores)) if select(i, 0)]
    #
    # return np.array(scores_mw), np.array(scores_gw)


def find_random_search_thresholds_with_constraints(
        scores, predicted_labels, groundtruth_labels, maximise_vals,
        constraint_vals, max_samples=100, quiet=False, ncpu=-1, full=True):
    """Perform a random grid search to find the best thresholds on `scores` in
    parallel.

    This method wraps `find_random_search_thresholds_with_constraints_discrete`
    and parallelizes it. For a full description of this, read the documentation
    of the aformentioned method. 

    See Also:
        - `find_random_search_threhsolds_with_constraint_discrete``

    Args:
        scores (dict): The test scores on which to perform the random search.
        predicted_labels (np.ndarray): The set of predictions to decide which
            'per-class' threshold to use.
        groundtruth_labels (np.ndarray): The groundtruth label for each object.
        maximise_vals: The metrics that should be maximised.
        constraint_vals: The metrics that are constrained.
        max_samples (int): The maximum number of random threshold combinations
            to try before settling for the best performance up to that point.
        quiet (bool): If True, logging will be disabled.
        ncpu (int): Number of cpus to use, if negative then we compute it as
            total_cpu + ncpu, if ncpu=1 then we do not parallelize, this is done 
            to avoid problems with nested parallelization

    Returns:
        dict: Set of thresholds for malware ('gw') and goodware ('gw') classes.

    """

    ncpu = mp.cpu_count() + ncpu if ncpu < 0 else ncpu

    if ncpu == 1:
        results, thresholds = find_random_search_thresholds_with_constraints_discrete(
            scores, predicted_labels, groundtruth_labels, maximise_vals,
            constraint_vals, max_samples, quiet, full)

        return thresholds

    samples = [max_samples // ncpu for _ in range(ncpu)]

    with mp.Pool(processes=ncpu) as pool:
        results = pool.starmap(find_random_search_thresholds_with_constraints_discrete,
                               zip(repeat(scores), repeat(predicted_labels), repeat(groundtruth_labels),
                                   repeat(maximise_vals), repeat(constraint_vals), samples, repeat(quiet),
                                   repeat(full)))
        results_list = [res[0] for res in results]
        thresholds_list = [res[1] for res in results]

    def resolve_keyvals(s):
        if isinstance(s, str):
            pairs = s.split(',')
            pairs = [x.split(':') for x in pairs]
            return {k: float(v) for k, v in pairs}
        return s

    maximise_vals = resolve_keyvals(maximise_vals)
    constraint_vals = resolve_keyvals(constraint_vals)

    best_maximised = {k: 0 for k in maximise_vals}
    best_constrained = {k: 0 for k in constraint_vals}
    best_thresholds, best_result = {}, {}

    for result, thresholds in zip(results_list, thresholds_list):
        if any([result[k] < constraint_vals[k] for k in constraint_vals]):
            continue

        if any([result[k] < best_maximised[k] for k in maximise_vals]):
            continue
        # Small boost for pessimism
        if all([result[k] == best_maximised[k] for k in maximise_vals]):
            if all([result[k] >= best_constrained[k] for k in constraint_vals]):
                best_maximised = {k: result[k] for k in maximise_vals}
                best_constrained = {k: result[k] for k in constraint_vals}
                best_thresholds = thresholds
                best_result = result

                if not quiet != quiet:
                    logging.info('New best: {}|{}\n    @ {} '.format(
                        format_opts(maximise_vals.keys(), result),
                        format_opts(constraint_vals.keys(), result),
                        best_thresholds))
                    report_results(d=best_result, full=full)
            continue

        # Big boost for optimism
        if any([result[k] > best_maximised[k] for k in maximise_vals]):
            best_maximised = {k: result[k] for k in maximise_vals}
            best_constrained = {k: result[k] for k in constraint_vals}
            best_thresholds = thresholds
            best_result = result

            if not quiet != quiet:
                logging.info('New best: {}|{} \n    @ {} '.format(
                    format_opts(maximise_vals.keys(), result),
                    format_opts(constraint_vals.keys(), result),
                    best_thresholds))
                report_results(d=best_result, full=full)

            continue

    if best_thresholds == {}:
        print('No appropriate threshold exists,\n'
              'try modifying the con_threshold limit or gamma and C,\n'
              'or increase the number of samples or modify the random_threshold()')
        best_thresholds = {'cred': {'mw': 1, 'gw': 1}}

    print(best_thresholds)
    return best_thresholds


def find_random_search_thresholds_with_constraints_discrete(
        scores, predicted_labels, groundtruth_labels, maximise_vals,
        constraint_vals, max_samples=100, quiet=False, full=True, stop_condition=3000):
    """Perform a random grid search to find the best thresholds on `scores`.

    `scores` expects a dictionary keyed by 'cred' and/or 'conf',
    with sub-dictionaries containing the thresholds for the mw and gw classes.

    Note that the keys of `scores` determine _which_ thresholding criteria will
    be enforced. That is, if only a 'cred' dictionary is supplied, thresholding
    will be enforced on cred-only and the same for 'conf'. Supplying cred and
    conf dictionaries will enforce the 'cred+conf' thresholding criteria (all
    thresholds will be applied).

    `maximise_vals` describes the metrics that should be maximised and their
    minimum acceptable values. It expects either a dictionary of metrics, or a
    string or comma separated metrics.

    `constrained_vals` describes the floors for metrics that a threshold must
    pass in order to be acceptable. The algorithm will also try to maximise
    these metrics if possible, although never at the expense of `maximise_vals`.

    Both `maximise_vals` and `constrained_vals` expect a dictionary of metrics
    and maximum acceptable values. Alternatively, arguments can be given in
    string form as comma-separated key:value pairs, for example,
    'key1:value1,key2:value2,key3:value3'.

    Concretely, any of the following are acceptable:

        > maximise_vals = {'f1': 0.95}
        > maximise_vals = 'f1_k:0.95'

        > constrained_vals = {'kept_pos_perc': 0.76, 'kept_neg_perc': 0.76}
        > constrained_vals = kept_pos_perc:0.76,kept_neg_perc:0.76

    For a list of possible metrics, see the keys in the dict produced by
    `get_performance_with_rejection()`. Note that the default objective
    function assumes that the provided metrics are in the interval [0,1].

    See Also:
        - `get_performance_with_rejection`

    Args:
        scores (dict): The test scores on which to perform the random search.
        predicted_labels (np.ndarray): The set of predictions to decide which
            'per-class' threshold to use.
        groundtruth_labels (np.ndarray): The groundtruth label for each object.
        maximise_vals: The metrics that should be maximised.
        constraint_vals: The metrics that are constrained.
        max_samples (int): The maximum number of random threshold combinations
            to try before settling for the best performance up to that point.
        quiet (bool): If True, logging will be disabled.

    Returns:
        dict: Set of thresholds for malware ('gw') and goodware ('gw') classes.

    """

    # as this method is called from multiprocessing, we want to make sure each
    # process has a different seed 
    seed = 0
    for l in os.urandom(10): seed += l
    np.random.seed(seed)

    def resolve_keyvals(s):
        if isinstance(s, str):
            pairs = s.split(',')
            pairs = [x.split(':') for x in pairs]
            return {k: float(v) for k, v in pairs}
        return s

    maximise_vals = resolve_keyvals(maximise_vals)
    constraint_vals = resolve_keyvals(constraint_vals)

    best_maximised = {k: 0 for k in maximise_vals}
    best_constrained = {k: 0 for k in constraint_vals}
    best_thresholds, best_result = {}, {}

    logging.info('Searching for threshold on calibration data...')

    stop_counter = 0

    for _ in tqdm(range(max_samples)):
        # Choose and package random thresholds
        thresholds = {}
        if 'cred' in scores:
            cred_thresholds = random_threshold(scores['cred'], predicted_labels)
            thresholds['cred'] = cred_thresholds
        if 'conf' in scores:
            conf_thresholds = random_threshold(scores['conf'], predicted_labels)
            thresholds['conf'] = conf_thresholds

        # Test with chosen thresholds
        result, h, g = test_with_rejection(
            thresholds, scores, groundtruth_labels, predicted_labels, full)

        # Check if any results exceed given constraints (e.g. too many rejects)
        if any([result[k] < constraint_vals[k] for k in constraint_vals]):
            if stop_counter > stop_condition:
                logging.info('Exceeded stop condition, terminating calibration search...')
                break

            stop_counter += 1
            continue

        if any([result[k] < best_maximised[k] for k in maximise_vals]):
            if stop_counter > stop_condition:
                logging.info('Exceeded stop condition, terminating calibration search...')
                break

            stop_counter += 1
            continue

        # Small boost for pessimism
        if all([result[k] == best_maximised[k] for k in maximise_vals]):
            if all([result[k] >= best_constrained[k] for k in constraint_vals]):
                best_maximised = {k: result[k] for k in maximise_vals}
                best_constrained = {k: result[k] for k in constraint_vals}
                best_thresholds = thresholds
                best_result = result

                print('\n[*] New best: {}|{}\n    @ {} '.format(
                    format_opts(maximise_vals.keys(), result),
                    format_opts(constraint_vals.keys(), result),
                    best_thresholds))
                if not quiet:
                    report_results(d=best_result, full=full)

            stop_counter = 0
            continue

        # Big boost for optimism
        if any([result[k] > best_maximised[k] for k in maximise_vals]):
            best_maximised = {k: result[k] for k in maximise_vals}
            best_constrained = {k: result[k] for k in constraint_vals}
            best_thresholds = thresholds
            best_result = result

            print('\n[*] New best: {}|{}\n    @ {} '.format(
                format_opts(maximise_vals.keys(), result),
                format_opts(constraint_vals.keys(), result),
                best_thresholds))
            if not quiet:
                report_results(d=best_result, full=full)

            stop_counter = 0
            continue

    if not bool(best_result):
        best_result = result

    return (best_result, best_thresholds)


def random_threshold(scores, predicted_labels):
    """Produce random thresholds over the given scores.

    Args:
        scores (dict): The test scores on which to produce a threshold.
        predicted_labels (np.ndarray): The set of predictions to decide which
            'per-class' threshold to use.

    Returns:
        dict: Set of thresholds for malware ('gw') and goodware ('gw') classes.

    """
    scores_mw, scores_gw = sort_by_predicted_label(
        scores, predicted_labels, np.array([]), 'all')

    mw_threshold = np.random.uniform(min(scores_mw), max(scores_mw))
    gw_threshold = np.random.uniform(min(scores_gw), max(scores_gw))
    #######################
    # print(mw_threshold, gw_threshold)
    return {'mw': mw_threshold, 'gw': gw_threshold}


def get_performance_with_rejection(y_true, y_pred, keep_mask, full=True):
    """Get test results, rejecting predictions based on a given keep mask.

    Args:
        y_true (np.ndarray): The groundtruth label for each object.
        y_pred (np.ndarray): The set of predictions to decide which 'per-class'
            threshold to use. Depending on the stage of conformal evaluation,
            this could be either the predicted or ground truth labels.
        keep_mask (np.ndarray): A boolean mask describing which elements to
            keep (True) or reject (False).
        full (bool): True if full statistics are required, False otherwise.
            False is computationally less expensive.

    Returns:
        dict: A dictionary of results for baseline, kept, and rejected metrics.

    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    d = {}

    total = len(y_pred)

    kept_total = sum(keep_mask)
    reject_total = total - kept_total

    d.update({'total': total,
              'kept_total': kept_total, 'reject_total': reject_total,
              'kept_total_perc': kept_total / total,
              'reject_total_perc': reject_total / total,
              })

    return d


def report_results(d, quiet=False, full=False):
    """Produce a textual report based on the given results.

    Args:
        d (dict): Results for baseline, kept, and rejected metrics.
        quiet (bool): Whether to also print the results to stdout.

    Returns:
        str: A textual report of the results.

    """
    report_str = ''

    def print_and_extend(report_line):
        nonlocal report_str
        if not quiet:
            cprint(report_line, 'yellow')
        report_str += report_line + '\n'

    s = '% kept elements(kt):   \t{:6d}/{:6d} = {:.1f}%, \t% rejected elements(rt):   \t{:6d}/{:6d} = {:.1f}%'.format(
        d['kept_total'], d['total'], d['kept_total'] / d['total'] * 100,
        d['reject_total'], d['total'], d['reject_total'] / d['total'] * 100
    )
    print_and_extend(s)

    return report_str


def format_opts(metrics, results):
    """Helper function for formatting the results of a list of metrics."""
    return (' {}: {:.4f} |' * len(metrics)).format(
        *[item for sublist in
          zip(metrics, [results[k] for k in metrics]) for
          item in sublist])
