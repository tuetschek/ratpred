#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import unicode_literals

import sys
import os
from argparse import ArgumentParser
from itertools import product

from tgen.logf import log_info
sys.path.insert(0, os.path.abspath('../'))  # add ratpred main directory to modules path
from ratpred.eval import Evaluator


def process_data(args):

    if args.subdirs:
        subdirs = next(os.walk(args.subdirs))[1]
        fnames = [os.path.join(args.subdirs, subdir, tsv)
                  for subdir, tsv in product(subdirs, args.tsvs)]
    else:
        fnames = args.tsvs
    evaler = Evaluator()
    for fname in fnames:
        evaler.append_from_tsv(fname)
    results = evaler.get_stats(hide_nans=False)
    log_info("Evaluation over file(s): %s" % ", ".join(fnames))
    for tc in sorted(results.keys()):
        log_info("%s Distance: %.3f (avg: %.3f, std: %.3f)" % (tc.upper(),
                                                               results[tc]['dist_total'],
                                                               results[tc]['dist_avg'],
                                                               results[tc]['dist_stddev']))
        log_info("%s MAE: %.3f, RMSE: %.3f" % (tc.upper(), results[tc]['mae'], results[tc]['rmse']))
        log_info("%s Accuracy: %.3f" % (tc.upper(), results[tc]['accuracy']))
        log_info("%s Pearson correlation: %.3f (p-value %.3f)" %
                 (tc.upper(), results[tc]['pearson'], results[tc]['pearson_pv']))
        log_info("%s Spearman correlation: %.3f (p-value %.3f)" %
                 (tc.upper(), results[tc]['spearman'], results[tc]['spearman_pv']))
        log_info("%s Pairwise rank accuracy: %.3f" % (tc.upper(), results[tc]['rank_acc']))
        log_info("%s Pairwise rank loss: %.3f (avg: %.3f)" %
                 (tc.upper(), results[tc]['rank_loss_total'], results[tc]['rank_loss_avg']))

if __name__ == '__main__':
    ap = ArgumentParser(description='Compute overall statistics from CV/rands runs.')
    ap.add_argument('-s', '--subdirs',
                    help='Look for the given TSV file name in all immediate subdirs ' +
                    'of this directory, issue error if not found.')
    ap.add_argument('tsvs', nargs='+',
                    help='TSV output files with predictions from the individual runs.')
    args = ap.parse_args()
    process_data(args)
