#!/usr/bin/env python
# -"- coding: utf-8 -"-


from __future__ import unicode_literals
import pandas as pd
from argparse import ArgumentParser
from tgen.data import DA

# Start IPdb on error in interactive mode
from tgen.debug import exc_info_hook
import sys
import itertools

sys.excepthook = exc_info_hook


COLUMN_MAP = {'inf': 'informativeness',
              'nat': 'naturalness',
              'qua': 'quality'}

COLUMNS = ['dataset',  # E2E always
           'system', 'system2',
           'mr',
           'orig_ref',  # will be empty
           'system_ref', 'system_ref2',
           'is_real',  # always 1
           'informativeness',  # only one of these will be set
           'naturalness',
           'quality']

SYSTEMS_MAP = {'baseline': 'TGen',
               'biao_zhang': 'Zhang',
               'chen shuang': 'Chen',
               'deriu jan milan v1': 'ZHAW1',
               'deriu jan milan v2': 'ZHAW2',
               'forge_v1': 'FORGe1',
               'forge_v2': 'FORGe3',
               'harvardnlp': 'Harv',
               'heng gong': 'Gong',
               'henry elder': 'Adapt',
               'sheffield_v1': 'Sheff1',
               'sheffield_v2': 'Sheff2',
               'shubham agarwal': 'NLE',
               'slug2slug': 'Slug',
               'slug2slug_alt': 'Slug-alt',
               'thomson reuters_v1': 'TR1',
               'thomson reuters_v2': 'TR2',
               'tnt-nlg_v1': 'TNT1',
               'tnt-nlg_v2': 'TNT2',
               'uit-dangnt': 'DANGNT',
               'ukp-tuda': 'TUDA'}


def convert(args):
    src = pd.read_csv(args.src_file, index_col=None, encoding='utf-8')
    data = []
    src_col = args.column
    trg_col = COLUMN_MAP[src_col[:3]]
    for _, src_inst in src.iterrows():
        mr = DA.parse_diligent_da(src_inst['mr']).to_cambridge_da_string()
        sys = [{'sys': src_inst['sys%d' % i],
                'ref': src_inst['ref%d' % i],
                'val': src_inst['%s%d' % (src_col, i)]} for i in xrange(1, 6)]

        for sys1, sys2 in itertools.combinations(sys, 2):
            if sys1['val'] < sys2['val']:  # without loss of generality
                sys1, sys2 = sys2, sys1
            if sys1['val'] == sys2['val']:  # ignore those that are equal
                continue
            trg_inst = {'dataset': 'E2E',
                        'system': SYSTEMS_MAP[sys1['sys']],
                        'system2': SYSTEMS_MAP[sys2['sys']],
                        'orig_ref': None,
                        'mr': mr,
                        'system_ref': sys1['ref'],
                        'system_ref2': sys2['ref'],
                        'is_real': 1,
                        'informativeness': None,
                        'naturalness': None,
                        'quality': None}
            trg_inst[trg_col] = 1
            data.append(trg_inst)

    df = pd.DataFrame(data)
    df.to_csv(args.out_file, columns=COLUMNS, sep=b"\t", index=False, encoding='UTF-8')


if __name__ == '__main__':
    ap = ArgumentParser(description='Convert E2E data to our format.')
    ap.add_argument('--column', '-d', type=str, help='Column to use (others will be NaN)', required=True)
    ap.add_argument('src_file', type=str, help='Path to source data file')
    ap.add_argument('out_file', type=str, help='Output TSV file')

    args = ap.parse_args()
    convert(args)
