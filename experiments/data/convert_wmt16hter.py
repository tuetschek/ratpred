#!/usr/bin/env python
# -"- coding: utf-8 -"-

import pandas as pd
import codecs
from argparse import ArgumentParser


def lines_to_list(file_name):
    with codecs.open(file_name, 'r', 'UTF-8') as fh:
        out = []
        for line in fh:
            out.append(line.strip())
    return out


def convert(args):
    src = lines_to_list(args.src_file)
    sys = lines_to_list(args.sys_file)
    ref = lines_to_list(args.ref_file)
    score = [float(score) for score in lines_to_list(args.score_file)]
    df = pd.DataFrame.from_dict({'mr': src,
                                 'system_ref': sys,
                                 'orig_ref': ref,
                                 'quality': score})
    df.to_csv(args.out_file, columns=['mr', 'system_ref', 'orig_ref', 'quality'],
              sep=b"\t", index=False, encoding='UTF-8')


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('src_file', type=str, help='Path to source data file')
    ap.add_argument('sys_file', type=str, help='Path to system output data file')
    ap.add_argument('ref_file', type=str, help='Path to reference (PE) data file')
    ap.add_argument('score_file', type=str, help='Path to score data file')
    ap.add_argument('out_file', type=str, help='Output TSV file')

    args = ap.parse_args()
    convert(args)
