#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import pandas as pd
import numpy as np
import os.path
from argparse import ArgumentParser

from tgen.logf import log_info


def get_data_parts(data, sizes):
    """Split the data into parts, given the parts' proportional sizes.
    @param data: the data to be split
    @param sizes: parts' sizes ratios
    @return a list of data parts
    """
    parts = []
    total_parts = sum(sizes)
    sizes = [int(round((part / float(total_parts)) * len(data))) for part in sizes]
    sizes[0] += len(data) - sum(sizes)  # 1st part takes the rounding error
    offset = 0
    for size in sizes:
        part = data.iloc[offset:offset + size, :]
        offset += size
        parts.append(part)
    return parts

def convert(args):
    """Main function, does the conversion, taking parsed command-line arguments.
    @param args: command-line arguments
    """

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

    if args.cv:  # for cross-validation, just pre-split the data to small parts (to be compiled later)
        cv_sizes = sizes
        sizes = [1] * sum(sizes)

    parts = get_data_parts(data, sizes)

    if args.devtest_crit:
        parts = [train_part] + parts

    if args.cv:  # for cross-validation, compile the data, repeating the parts with a shift
        cv_parts = []
        cv_labels = []
        for offset in xrange(len(sizes)):
            os.mkdir(os.path.join(args.output_dir, 'cv%02d' % offset))
            cur_parts = parts[offset:] + parts[:offset]
            for cv_size, cv_label in zip(cv_sizes, labels):
                cv_parts.append(pd.concat(cur_parts[:cv_size]))
                cur_parts = cur_parts[cv_size:]
                cv_labels.append(os.path.join('cv%02d' % offset, cv_label))
        labels = cv_labels
        parts = cv_parts

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
    ap.add_argument('-v', '--cv', action='store_true',
                    help='Create cross-validation files (as many parts as ' +
                    'there are in the data split ratio)?')
    ap.add_argument('input_file', type=str, help='Path to the input file')
    ap.add_argument('output_dir', type=str,
                    help='Output directory (where train,devel,test TSV will be created)')

    np.random.seed(1206)

    args = ap.parse_args()
    convert(args)
