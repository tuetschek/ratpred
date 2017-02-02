#!/usr/bin/env python
# -"- coding: utf-8 -"-


from __future__ import unicode_literals
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from tgen.data import DA

# Start IPdb on error in interactive mode
from tgen.debug import exc_info_hook
import sys
sys.excepthook = exc_info_hook


COLUMNS = ['dataset', 'system', 'orig_ref', 'Bleu_4',
           'Bleu_3', 'CIDEr', 'Bleu_1', 'Bleu_2', 'ROUGE_L',
           'LEPOR', 'METEOR', 'mr', 'is_real', 'TER', 'system_ref',
           'NIST', 'judge', 'informativeness', 'naturalness',
           'quality', 'judge_id']

COLUMN_MAP = {'nl.utterance': ['orig_ref', 'system_ref'],
              'informativeness': 'informativeness',
              'naturalness': 'naturalness',
              'phrasing': 'quality', }


def convert(args):
    src = pd.read_csv(args.src_file, index_col=None, encoding='utf-8')
    df = pd.DataFrame(index=np.arange(len(src)), columns=COLUMNS)
    for src_col, trg_col in COLUMN_MAP.iteritems():
        if isinstance(trg_col, list):
            for trg_col_ in trg_col:
                df[trg_col_] = src[src_col]
        else:
            df[trg_col] = src[src_col]
    df['mr'] = [DA.parse_diligent_da(da).to_cambridge_da_string() for da in src['mr']]
    df['is_real'] = np.ones(len(src), dtype=np.int32)
    df['dataset'] = ['INLG'] * len(src)
    df['system'] = ['human'] * len(src)
    df.to_csv(args.out_file, columns=COLUMNS, sep=b"\t", index=False, encoding='UTF-8')


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('src_file', type=str, help='Path to source data file')
    ap.add_argument('out_file', type=str, help='Output TSV file')

    args = ap.parse_args()
    convert(args)
