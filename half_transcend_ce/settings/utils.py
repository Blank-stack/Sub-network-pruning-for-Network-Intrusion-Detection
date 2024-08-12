# -*- coding: utf-8 -*-

"""
utils.py
~~~~~~~~

Helper functions for setting up the environment.

"""
import argparse
import logging
import multiprocessing as mp
import sys
from pprint import pformat

from termcolor import colored


def configure_logger():
    class SpecialFormatter(logging.Formatter):
        FORMATS = {
            logging.DEBUG: logging._STYLES['{'][0](colored("[*] {message}", 'blue')),
            logging.INFO: logging._STYLES['{'][0](colored("[*] {message}", 'cyan')),
            logging.WARNING: logging._STYLES['{'][0](colored("[!] {message}", 'yellow')),
            logging.ERROR: logging._STYLES['{'][0](colored("[!] {message}", 'red')),
            logging.CRITICAL: logging._STYLES['{'][0](colored("[!] {message}", 'white', 'on_red')),
            'DEFAULT': logging._STYLES['{'][0]("[ ] {message}")}

        def format(self, record):
            self._style = self.FORMATS.get(record.levelno,
                                           self.FORMATS['DEFAULT'])
            return logging.Formatter.format(self, record)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(SpecialFormatter())
    logging.root.addHandler(handler)
    logging.root.setLevel(logging.DEBUG)


def parse_args():
    """Parse the command line configuration for a particular run.

    See Also:
        - `data.load_features`
        - `thresholding.find_quartile_thresholds`
        - `thresholding.find_random_search_thresholds`
        - `thresholding.sort_by_predicted_label`
        - `thresholding.get_performance_with_rejection`

    Returns:
        argparse.Namespace: A set of parsed arguments.

    """
    p = argparse.ArgumentParser()

    # Dataset options
    p.add_argument('--train', default='KDDTrain.csv',
                   help='The training set to use.')
    p.add_argument('--test', default='KDDTest.csv',
                   help='The testing set to use.')

    # Thresholding options
    p.add_argument('-t', '--thresholds', default='quartiles',
                   choices=['quartiles', 'random-search', 'constrained-search', 'full-search'],
                   help='The type of thresholds to use.')

    p.add_argument('-c', '--criteria', default='cred+conf',
                   choices=['cred', 'conf', 'cred+conf'],
                   help='The p-values to threshold on.')

    # Sub-arguments for --thresholds=quartiles
    p.add_argument('--q-consider', default='correct',
                   choices=['correct', 'incorrect', 'all'],
                   help='Which predictions to select quartiles from.')

    # Sub-arguments for --thresholds=constrained-search
    p.add_argument('--rs-samples', type=int, default=300,
                   help='The number of sample selections to make.')
    p.add_argument('--cs-max', default='f1_k:0.1',
                   help='The performance metric(s) to maximise. '
                        'Comma separated key:value pairs (e.g., "f1_k:0.99")')
    # p.add_argument('--cs-con', default='kept_neg_perc:0.6,tpr_k:0.9,tnr_k:0.4,kept_pos_perc:0.4',
    p.add_argument('--cs-con', default='kept_neg_perc:0.85,tpr_k:0.8,tnr_k:0.9,kept_pos_perc:0.7',
                   help='The performance metric(s) to constrain. '
                        'Comma separated key:value pairs'
                        ' (e.g., "kept_total_perc:0.75")'
                        ' (e.g., "kept_neg_perc:0.8,reject_pos_perc:0.7")')
    args = p.parse_args()

    # logging.warning('Running with configuration:\n' + pformat(vars(args)))
    return args
