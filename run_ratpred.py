#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import unicode_literals

import sys
from argparse import ArgumentParser

from flect.config import Config
from tgen.logf import log_info
from tgen.debug import exc_info_hook

from ratpred.futil import read_data
from ratpred.predictor import RatingPredictor


sys.excepthook = exc_info_hook


def main():
    ap = ArgumentParser()
    ap.add_argument('-t', '--target', type=str,
                    help='Target column (default: quality)', default='quality')
    ap.add_argument('-p', '--training-portion', type=float,
                    help='Part of data used for training (rest is test)', default=0.8)
    ap.add_argument('-d', '--delex-slots', type=str,
                    help='Comma-separated list of slots to delexicalize', default='')
    ap.add_argument('config_file', type=str, help='Path to the configuration file')
    ap.add_argument('data_file', type=str, help='Path to the data TSV file')

    args = ap.parse_args()

    log_info("Loading data...")
    cfg = Config(args.config_file)
    delex_slots = set(args.delex_slots.split(',') if args.delex_slots else [])
    inputs, targets = read_data(args.data_file, args.target, delex_slots)

    train_len = int(len(inputs) * args.training_portion)
    train_insts = inputs[:train_len]
    train_targets = targets[:train_len]
    test_insts = inputs[train_len:]
    test_targets = targets[train_len:]

    log_info("Creating predictor...")
    rp = RatingPredictor(cfg)
    log_info("Training on %d instances..." % len(train_insts))
    rp.train(train_insts, train_targets)

    log_info("Testing on %d instances..." % len(test_insts))
    dist = rp.evaluate(test_insts, test_targets)
    log_info("Distance: %.3f" % dist)


if __name__ == '__main__':
    main()
