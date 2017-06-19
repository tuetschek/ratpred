#!/usr/bin/env python
# -"- coding: utf-8 -"-

from argparse import ArgumentParser
import pandas as pd
import numpy as np
import scipy.stats
import os


def main(args):
    data_a = pd.DataFrame.from_csv(args.pred_file_a, index_col=None, sep='\t')
    data_b = pd.DataFrame.from_csv(args.pred_file_b, index_col=None, sep='\t')
    assert(len(data_a) == len(data_b))
    data_a = data_a.sort_values(by=['mr', 'system_output']).reset_index()
    data_b = data_b.sort_values(by=['mr', 'system_output']).reset_index()
    ref = np.array(data_a['human_rating'])
    pred_a = np.array(data_a['system_rating'])
    pred_b = np.array(data_b['system_rating'])
    for corr in [scipy.stats.pearsonr, scipy.stats.spearmanr]:
        c12, _ = corr(ref, pred_a)
        c13, _ = corr(ref, pred_b)
        c23, _ = corr(pred_a, pred_b)
        print corr.__name__, c12, c13, c23, len(data_a)
        if c12 < c13:  # swap that 1st is always bigger
            c12, c13 = c13, c12
        os.system("R --no-save --args %f %f %f %d < williams.R | grep '^P-value'" % (c12, c13, c23, len(data_a)))


if __name__ == '__main__':
    ap = ArgumentParser(description='Williams correlation test ' +
                        '(a wrapper for the R script by Y. Graham)')
    ap.add_argument('pred_file_a', type=str, help='1st file with predictions to compare')
    ap.add_argument('pred_file_b', type=str, help='2nd file with predictions to compare')
    main(ap.parse_args())
