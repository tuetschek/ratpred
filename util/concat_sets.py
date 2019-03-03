#!/usr/bin/env python
# -"- coding: utf-8 -"-

from argparse import ArgumentParser
import pandas as pd


def main(args):
    data = [pd.read_csv(fname, index_col=None, sep='\t') for fname in args.src_files]
    data = pd.concat(data).reset_index(drop=True)
    data.to_csv(args.trg_file, index=None, sep='\t')


if __name__ == '__main__':
    ap = ArgumentParser(description='Concatenate TSV datasets ' +
                        '(useful for significance tests with multiple initializations)')
    ap.add_argument('src_files', type=str, nargs='+', help='Source sets to concat')
    ap.add_argument('trg_file', type=str, help='Output concatenated set')
    main(ap.parse_args())
