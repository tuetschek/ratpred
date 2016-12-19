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

from ratpred.futil import read_data
from ratpred.predictor import RatingPredictor


sys.excepthook = exc_info_hook


def train(args):

    if args.random_seed:  # set random seed if needed
        rnd.seed(args.random_seed)

    log_info("Loading configuration from %s..." % args.config_file)
    cfg = Config(args.config_file)

    log_info("Initializing...")
    rp = RatingPredictor(cfg)
    log_info("Training...")
    rp.train(args.train_data, valid_data_file=args.valid_data, data_portion=args.training_portion)
    log_info("Saving model to %s..." % args.model_file)
    rp.save_to_file(args.model_file)


def test(args):

    log_info("Loading model from %s..." % args.model_file)
    rp = RatingPredictor.load_from_file(args.model_file)

    log_info("Loading test data from %s..." % args.test_data)
    inputs, targets = read_data(args.test_data, rp.target_col, rp.delex_slots)

    log_info("Rating %d instances..." % len(inputs))
    dist, std, acc = rp.evaluate(inputs, targets, args.write_outputs)
    log_info("Distance: %.3f (avg: %.3f, std: %.3f)" % (dist, dist / len(inputs), std))
    log_info("Accuracy: %.3f" % acc)


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
