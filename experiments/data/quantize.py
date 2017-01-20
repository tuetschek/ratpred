#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import pandas as pd
import numpy as np
from argparse import ArgumentParser

from tgen.logf import log_info


def normalize(col):
    lo = min(col)
    hi = max(col)
    return [(val - lo) / float(hi - lo) for val in col]


def quantize(col, num_vals):
    step = 1 / float(num_vals)
    return [min(float(num_vals - 1), np.floor(val / step)) for val in col]


def flip(col):
    return [1.0 - val for val in col]


def convert(args):
    """Main function, does the conversion, taking parsed command-line arguments.
    @param args: command-line arguments
    """

    log_info("Loading %s..." % args.input_file)
    df = pd.read_csv(args.input_file)
    cols = [col for col in df.columns if col not in ['judge', 'judge_id']]
    log_info("Contains %d instances." % len(df))

    if args.median:
        log_info("Computing medians...")
        group_cols = list(set(df.columns) - set(['informativeness', 'naturalness',
                                                 'quality', 'judge', 'judge_id']))
        df = df.groupby(group_cols, as_index=False).median()

    if args.normalize:
        log_info("Normalizing '%s'..." % args.normalize.split(','))
        for col in args.normalize.split(','):
            df[col] = normalize(df[col])

    if args.add_random:
        log_info("Adding random column '%s'..." % args.add_random)
        df[args.add_random] = [np.random.random() for _ in xrange(len(df))]
        cols = cols[:-3] + [args.add_random] + cols[-3:]

    if args.flip:
        log_info("Flipping '%s'..." % args.flip.split(','))
        for col in args.flip.split(','):
            df[col] = flip(df[col])

    if args.quantize:
        log_info("Quantizing '%s'..." % args.quantize.split(','))
        for col in args.quantize.split(','):
            df[col] = [val / 2.0 + 1 for val in quantize(df[col], 11)]

    log_info("Writing %s..." % args.output_file)
    df.to_csv(args.output_file, index=False, columns=cols)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('-m', '--median', action='store_true',
                    help='Group human ratings and use medians')
    ap.add_argument('-n', '--normalize', type=str, help='List of columns to normalize')
    ap.add_argument('-q', '--quantize', type=str,
                    help='List of columns to quantize (assumes normalized data)')
    ap.add_argument('-f', '--flip', type=str,
                    help='List of columns to flip (assumes normalized data)')
    ap.add_argument('-r', '--add-random', type=str,
                    help='Add random column with the given name')
    ap.add_argument('input_file', type=str, help='Path to the input file')
    ap.add_argument('output_file', type=str, help='Output file path')

    np.random.seed(1206)

    args = ap.parse_args()
    convert(args)
