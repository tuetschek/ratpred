#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import sys
from argparse import ArgumentParser

import pandas as pd
import numpy.random as rnd

from tgen.debug import exc_info_hook
sys.excepthook = exc_info_hook


def pairwise_bootstrap(data_a, data_b, label_a, label_b, iters):

    assert(len(data_a) == len(data_b))

    a_better, b_better, ties = 0, 0, 0
    for i in xrange(iters):
        sample = rnd.randint(0, len(data_a), len(data_a))
        sum_a_good = sum(data_a[i] for i in sample)
        sum_b_good = sum(data_b[i] for i in sample)

        if sum_a_good > sum_b_good:
            a_better += 1
        elif sum_b_good > sum_a_good:
            b_better += 1
        else:
            ties += 1

    print (('%s better: %d (%2.2f) | %s better: %d (%2.2f) | ties: %d (%2.2f)') %
           (label_a, a_better, float(a_better) / iters * 100,
            label_b, b_better, float(b_better) / iters * 100,
            ties, float(ties) / iters * 100,))


def main(args):
    rnd.seed(1206)
    data_a = pd.read_csv(args.results_file_a, encoding='UTF-8', sep=b'\t', index_col=None)
    data_b = pd.read_csv(args.results_file_b, encoding='UTF-8', sep=b'\t', index_col=None)

    pairwise_bootstrap(list(data_a[args.target_column + '_rank_ok']),
                       list(data_b[args.target_column + '_rank_ok']),
                       args.results_file_a, args.results_file_b,
                       args.bootstrap_iters)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('-b', '--bootstrap-iters', type=int, default=1000)
    ap.add_argument('-t', '--target-column', type=str, default='quality')
    ap.add_argument('results_file_a', type=str)
    ap.add_argument('results_file_b', type=str)
    args = ap.parse_args()

    main(args)
