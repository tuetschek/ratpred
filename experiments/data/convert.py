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
                                                  'quality', 'judge']))
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
    total_parts = sum(sizes)
    sizes = [int(round((part / float(total_parts)) * len(data))) for part in sizes]
    sizes[0] += len(data) - sum(sizes)  # training data take the rounding error
    offset = 0
    for label, size in zip(labels, sizes):
        # select the part
        # pandas doesn't respect python conventions, so we need to use -1
        part = data.iloc[offset:offset + size - 1,:]
        # write the output
        log_info("Writing part %s (size %d)..." % (label, size))
        part.to_csv(os.path.join(args.output_dir, label + '.tsv'), sep=b"\t", index=False)
        offset += size
    log_info("Done.")


if __name__ == '__main__':
    ap = ArgumentParser()
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

