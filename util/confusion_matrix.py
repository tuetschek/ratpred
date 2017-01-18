#!/usr/bin/env python
# -"- coding: utf-8 -"-

"""
Build a confusion matrix and print it as CSV, given a TSV file with classification
"""

from __future__ import unicode_literals

import pandas as pd
import sklearn.metrics
import numpy as np
from argparse import ArgumentParser


def add_labels(matrix, labels):
    return [[''] + labels] + [[label] + row for label, row in zip(labels, matrix.tolist())]


def process_file(data_file):

    df = pd.DataFrame.from_csv(data_file, index_col=None, sep='\t')
    human = [str(val) for val in df['human_rating']]
    system = [str(val) for val in df['system_rating']]
    labels = sorted(list(set(human).union(set(system))))

    M = sklearn.metrics.confusion_matrix(human, system, labels=labels)
    M_perc = np.array([[0 if sum(row) == 0 else float(x) / sum(row) for x in row] for row in M])

    M = add_labels(M, labels)
    M_perc = add_labels(M_perc, labels)

    out = pd.DataFrame(M + [[]] + M_perc)
    print out.to_csv(header=False,index=False)


if __name__ == '__main__':

    ap = ArgumentParser()
    ap.add_argument('classif_file', type=str, help='TSV file with classification')

    args = ap.parse_args()
    process_file(args.classif_file)
