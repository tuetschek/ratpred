#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import pandas as pd
import numpy as np
import os.path
from argparse import ArgumentParser

from tgen.logf import log_info

def convert(args):

    log_info("Loading %s..." % args.input_file)
    data = pd.read_csv(args.input_file)
    log_info("Contains %d instances." % len(data))
    if args.median:
        log_info("Computing medians...")
        group_cols = list(set(data.columns) - set(['informativeness', 'naturalness',
                                                  'quality', 'judge', 'judge_id']))
        data = data.groupby(group_cols, as_index=False).median()
    if args.concat_refs:
        log_info("Concatenating all references for the same outputs...")
        group_cols = list(set(data.columns) - set(['orig_ref']))
        data = data.groupby(group_cols, as_index=False).agg(lambda vals: ' <|> '.join(vals))
    if args.shuffle:
        log_info("Shuffling...")
        data = data.iloc[np.random.permutation(len(data))]
        data.reset_index(drop=True)
    sizes = [int(part) for part in args.ratio.split(':')]
    labels = args.labels.split(':')

    if args.devtest_crit:
        # select dev/test data based on a criterion
        crit_col, crit_val = args.devtest_crit.split('=')
        train_part = data[data[crit_col] != crit_val]  # training data is everything else
        data = data[data[crit_col] == crit_val]  # dev+test data have the criterion
        sizes = sizes[1:]  # training size does not matter (everything not fulfilling the criterion)

    # split the data into parts
    parts = []
    total_parts = sum(sizes)
    sizes = [int(round((part / float(total_parts)) * len(data))) for part in sizes]
    sizes[0] += len(data) - sum(sizes)  # 1st part take the rounding error
    offset = 0
    for size in sizes:
        part = data.iloc[offset:offset + size,:]
        offset += size
        parts.append(part)

    if args.devtest_crit:
        parts = [train_part] + parts

    for label, part in zip(labels, parts):
        # write the output
        log_info("Writing part %s (size %d)..." % (label, len(part)))
        part.to_csv(os.path.join(args.output_dir, label + '.tsv'), sep=b"\t", index=False)
    log_info("Done.")


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('-d', '--devtest-crit', type=str, default=None,
                    help='A criterion (column=val) for selecting devel/test examples')
    ap.add_argument('-r', '--ratio', type=str, default='3:1:1',
                    help='Train-devel-test split ratio')
    ap.add_argument('-l', '--labels', type=str, default='train:devel:test',
                    help='Train-devel-test labels')
    ap.add_argument('-s', '--shuffle', action='store_true',
                    help='Shuffle data before dividing?')
    ap.add_argument('-m', '--median', action='store_true',
                    help='Group human ratings and use medians')
    ap.add_argument('-c', '--concat_refs', action='store_true',
                    help='Join and concatenate all references?')
    ap.add_argument('input_file', type=str, help='Path to the input file')
    ap.add_argument('output_dir', type=str,
                    help='Output directory (where train,devel,test TSV will be created)')

    np.random.seed(1206)

    args = ap.parse_args()
    convert(args)


