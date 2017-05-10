#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import unicode_literals

import sys
import os
from argparse import ArgumentParser

from tgen.logf import log_info
sys.path.insert(0, os.path.abspath('../'))  # add ratpred main directory to modules path
from ratpred.eval import Evaluator


def process_data(args):
    evaler = Evaluator()
    for fname in args.tsvs:
        evaler.append_from_tsv(fname)
    results = evaler.get_stats()
    log_info("Distance: %.3f (avg: %.3f, std: %.3f)" % (results['dist_total'],
                                                        results['dist_avg'],
                                                        results['dist_stddev']))
    log_info("Accuracy: %.3f" % results['accuracy'])
    log_info("Pearson correlation: %.3f (p-value %.3f)" %
             (results['pearson'], results['pearson_pv']))
    log_info("Spearman correlation: %.3f (p-value %.3f)" %
             (results['spearman'], results['spearman_pv']))


if __name__ == '__main__':
    ap = ArgumentParser(description='Compute overall statistics from CV/rands runs.')
    ap.add_argument('tsvs', nargs='+',
                    help='TSV output files with predictions from the individual runs')
    args = ap.parse_args()
    process_data(args)