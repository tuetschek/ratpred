#!/usr/bin/env python
# -"- coding: utf-8 -"-

import pandas as pd
import codecs
from argparse import ArgumentParser
from tgen.data import DA


def lines_to_list(file_name):
    with codecs.open(file_name, 'r', 'UTF-8') as fh:
        out = []
        for line in fh:
            out.append(line.strip())
    return out


def convert(args):
    src = lines_to_list(args.src_file)
    if args.das:
        src = [DA.parse(da_text).to_cambridge_da_string() for da_text in src]
    ref = lines_to_list(args.ref_file)
    columns = ['mr', 'orig_ref']
    df = pd.DataFrame.from_dict({'mr': src, 'orig_ref': ref})

    if args.system_output:
        sys = lines_to_list(args.system_output)
        df['system_ref'] = sys
        columns.append('system_ref')

    if args.score:
        score = [float(score) for score in lines_to_list(args.score)]
        df['quality'] = score
        columns.append('quality')

    df.to_csv(args.out_file, columns=columns, sep=b"\t", index=False, encoding='UTF-8')


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('-d', '--das', action='store_true',
                    help='Treat sources as Alex-style DAs, store Cambridge-style DAs' +
                    'on the output')
    ap.add_argument('-o', '--system-output', type=str, default=None,
                    help='Path to system output data file')
    ap.add_argument('-s', '--score', type=str, help='Path to score data file')
    ap.add_argument('src_file', type=str, help='Path to source data file')
    ap.add_argument('ref_file', type=str, help='Path to reference/post-edit data file')
    ap.add_argument('out_file', type=str, help='Output TSV file')

    args = ap.parse_args()
    convert(args)
