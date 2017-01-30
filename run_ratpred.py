#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import unicode_literals

import sys
from argparse import ArgumentParser

from flect.config import Config
from tgen.logf import log_info, set_debug_stream
from tgen.debug import exc_info_hook
from tgen.futil import file_stream
from tgen.rnd import rnd

from ratpred.predictor import RatingPredictor


sys.excepthook = exc_info_hook


def train(args):

    if args.random_seed:  # set random seed if needed
        rnd.seed(args.random_seed)

    log_info("Loading configuration from %s..." % args.config_file)
    cfg = Config(args.config_file)

    log_info("Initializing...")
    rp = RatingPredictor(cfg)
    if args.tensorboard_dir_id is not None:
        tb_dir, run_id = args.tensorboard_dir_id.split(':', 1)
        rp.set_tensorboard_logging(tb_dir, run_id)
    log_info("Training...")
    rp.train(args.train_data, valid_data_file=args.valid_data,
             data_portion=args.training_portion, model_fname=args.model_file)
    log_info("Saving model to %s..." % args.model_file)
    rp.save_to_file(args.model_file)


def test(args):

    rp = RatingPredictor.load_from_file(args.model_file)

    log_info("Loading test data from %s..." % args.test_data)
    inputs, targets = rp.load_data(args.test_data)

    log_info("Rating %d instances..." % len(inputs))
    results = rp.evaluate(inputs, targets, args.write_outputs)
    log_info("Distance: %.3f (avg: %.3f, std: %.3f)" % (results['dist_total'],
                                                        results['dist_avg'],
                                                        results['dist_stddev']))
    log_info("Accuracy: %.3f" % results['accuracy'])
    log_info("Pearson correlation: %.3f (p-value %.3f)" %
             (results['pearson'], results['pearson_pv']))
    log_info("Spearman correlation: %.3f (p-value %.3f)" %
             (results['spearman'], results['spearman_pv']))


def main():
    ap = ArgumentParser()
    ap.add_argument('-d', '--debug-output', help='Path to debugging output file', type=str)

    subp = ap.add_subparsers()

    ap_train = subp.add_parser('train', help='Train a new rating predictor')
    ap_train.add_argument('-p', '--training-portion', type=float,
                          help='Part of data used for traing', default=1.0)
    ap_train.add_argument('-r', '--random-seed', type=str,
                          help='String to use as a random seed', default=None)
    ap_train.add_argument('-v', '--valid-data', type=str,
                          help='Path to validation data file', default=None)
    ap_train.add_argument('-t', '--tensorboard-dir-id', default=None,
                          help='Colon-separated path_to_tensorboard_logdir:run_id')
    ap_train.add_argument('config_file', type=str, help='Path to the configuration file')
    ap_train.add_argument('train_data', type=str, help='Path to the training data TSV file')
    ap_train.add_argument('model_file', type=str, help='Path where to store the predictor model')

    ap_test = subp.add_parser('test', help='Test a trained predictor on given data')
    ap_test.add_argument('-w', '--write-outputs', type=str,
                         help='Path to a prediction output file (not written when empty)',
                         default=None)
    ap_test.add_argument('model_file', type=str, help='Path to a trained predictor model')
    ap_test.add_argument('test_data', type=str, help='Path to the test data TSV file')

    args = ap.parse_args()
    if args.debug_output:
        ds = file_stream(args.debug_output, mode='w')
        set_debug_stream(ds)

    if hasattr(args, 'train_data'):
        train(args)
    else:
        test(args)


if __name__ == '__main__':
    main()
