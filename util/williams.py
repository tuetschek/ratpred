#!/usr/bin/env python
# -"- coding: utf-8 -"-
#
# Wrapper for https://github.com/ygraham/nlp-williams

from argparse import ArgumentParser
import pandas as pd
import numpy as np
import scipy.stats
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def main(args):
    print
    print args.pred_file_a, args.pred_file_b
    data_a = pd.read_csv(args.pred_file_a, index_col=None, sep='\t')
    data_b = pd.read_csv(args.pred_file_b, index_col=None, sep='\t')
    assert(len(data_a) == len(data_b))
    # ensure the order of instances is the same
    data_a = data_a.sort_values(by=['mr', 'system_output', args.target + '_human_rating']).reset_index()
    data_b = data_b.sort_values(by=['mr', 'system_output', args.target + '_human_rating']).reset_index()
    # get the relevant columns
    ref = np.array(data_a[args.target + '_human_rating'])
    pred_a = np.array(data_a[args.target + '_system_rating'])
    pred_b = np.array(data_b[args.target + '_system_rating'])
    for corr in [scipy.stats.pearsonr, scipy.stats.spearmanr]:
        c12, _ = corr(ref, pred_a)
        c13, _ = corr(ref, pred_b)
        c23, _ = corr(pred_a, pred_b)
        print corr.__name__, c12, c13, c23, len(data_a)
        if c12 < c13:  # swap so that 1st is always bigger
            print 'SWAPPING A-B'
            c12, c13 = c13, c12
        os.system("R --no-save --args %f %f %f %d < %s/williams.R | grep '^P-value'" %
                  (c12, c13, c23, len(data_a), SCRIPT_DIR))


if __name__ == '__main__':
    ap = ArgumentParser(description='Williams correlation test ' +
                        '(a wrapper for the R script by Y. Graham)')
    ap.add_argument('-t', '--target', type=str, default='quality',
                    help='Target column (default: quality)')
    ap.add_argument('pred_file_a', type=str, help='1st file with predictions to compare')
    ap.add_argument('pred_file_b', type=str, help='2nd file with predictions to compare')
    main(ap.parse_args())
