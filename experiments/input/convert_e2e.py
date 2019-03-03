#!/usr/bin/env python
# -"- coding: utf-8 -"-


from __future__ import unicode_literals
import pandas as pd
from argparse import ArgumentParser
from tgen.data import DA
from tgen.logf import log_info

# Start IPdb on error in interactive mode
from tgen.debug import exc_info_hook
import sys
import itertools
import random
import os
import codecs
import pprint
from collections import Counter

from convert import create_fake_data, create_fake_pairs

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


def get_sys_outputs(data):
    """Get instances with individual system outputs (regardless of pairs)."""
    sys_outputs = {}
    mrs = {}
    for inst in data:
        mrs[inst['mr']] = inst['delex_mr']
        sys_outputs[(inst['mr'], inst['system'])] = inst['system_ref']
        sys_outputs[(inst['mr'], inst['system2'])] = inst['system_ref2']

    sys_outs_list = []
    for (mr, sys_name), output in sys_outputs.iteritems():
        sys_outs_list.append({'dataset': 'E2E',
                              'mr': mr,
                              'delex_mr': mrs[mr],
                              'system': sys_name,
                              'system_ref': None,
                              'orig_ref': output,
                              'informativeness': None,
                              'naturalness': None,
                              'quality': None,
                              'is_real': 0})
    log_info('Using %d different system outputs to create fake pairs.' % len(sys_outs_list))
    return sys_outs_list


def read_system_training_data(filename):
    insts = []
    for inst in pd.read_csv(filename, index_col=None, encoding='UTF-8').to_dict('records'):
        insts.append({'dataset': 'E2E',
                      'mr': DA.parse_diligent_da(inst['mr']).to_cambridge_da_string(),
                      'delex_mr': DA.parse_diligent_da(inst['mr']).get_delexicalized(set(['name', 'near'])).to_cambridge_da_string(),
                      'system': 'HUMAN',
                      'system_ref': None,
                      'orig_ref': inst['ref'],
                      'informativeness': None,
                      'naturalness': None,
                      'quality': None,
                      'is_real': 0})
    log_info("Using %d different training human references to create fake pairs" % len(insts))
    return insts


def convert(args):
    src = pd.read_csv(args.src_file, index_col=None, encoding='utf-8')
    data = []
    src_col = args.column
    trg_col = COLUMN_MAP[src_col[:3]]
    unique_mrs = set()

    for _, src_inst in src.iterrows():
        mr = DA.parse_diligent_da(src_inst['mr']).to_cambridge_da_string()
        delex_mr = DA.parse_diligent_da(src_inst['mr']).get_delexicalized(set(['name', 'near'])).to_cambridge_da_string()
        unique_mrs.add(delex_mr)
        syss = [{'sys': src_inst['sys%d' % i],
                 'ref': src_inst['ref%d' % i],
                 'val': src_inst['%s%d' % (src_col, i)]} for i in xrange(1, 6)]

        for sys1, sys2 in itertools.combinations(syss, 2):
            if sys1['val'] < sys2['val']:  # without loss of generality
                sys1, sys2 = sys2, sys1
            if sys1['val'] == sys2['val']:  # ignore those that are equal
                continue
            trg_inst = {'dataset': 'E2E',
                        'system': SYSTEMS_MAP[sys1['sys']],
                        'system2': SYSTEMS_MAP[sys2['sys']],
                        'orig_ref': None,
                        'mr': mr,
                        'delex_mr': delex_mr,
                        'system_ref': sys1['ref'],
                        'system_ref2': sys2['ref'],
                        'is_real': 1,
                        'informativeness': None,
                        'naturalness': None,
                        'quality': None}
            trg_inst[trg_col] = 1
            data.append(trg_inst)

    unique_mrs = sorted(list(unique_mrs))
    random.shuffle(unique_mrs)

    part_sizes = [int(p) for p in args.ratio.split(':')]
    part_sizes = [int(round(p * len(unique_mrs) / float(sum(part_sizes)))) for p in part_sizes]
    part_sizes[0] = len(unique_mrs) - sum(part_sizes[1:])
    part_labels = args.labels.split(':')
    part_start = 0
    log_info('Data sizes in MRs: %s' % ':'.join([str(p) for p in part_sizes]))

    # remove ambiguous instances
    if args.unambiguous:
        occs = Counter([(inst['mr'], inst['system'], inst['system2']) for inst in data])
        ambig = set()
        for mr, sys1, sys2 in occs.iterkeys():
            if occs.get((mr, sys2, sys1), 0) == occs[(mr, sys1, sys2)]:
                ambig.add((mr, sys1, sys2))

        uniq_data = []
        used_insts = set()
        for inst in data:
            mr, sys1, sys2 = inst['mr'], inst['system'], inst['system2']
            if (mr, sys1, sys2) in ambig or (mr, sys1, sys2) in used_insts:
                continue
            uniq_data.append(inst)
            used_insts.add((mr, sys1, sys2))
        data = uniq_data

    # mark down the configuration
    with codecs.open(os.path.join(args.out_path, 'config'), 'wb', encoding='UTF-8') as fh:
        fh.write(pprint.pformat(vars(args), indent=4, width=100))

    # split the output
    for part_no, (part_size, part_label) in enumerate(zip(part_sizes, part_labels)):
        part_mrs = set(unique_mrs[part_start: part_start + part_size])
        part_data = [inst for inst in data if inst['delex_mr'] in part_mrs]

        if args.shuffle:
            random.shuffle(part_data)

        part_df = pd.DataFrame(part_data)

        if part_no == 0 and args.fake_data:
            # create fake data
            indiv_sys_outputs = get_sys_outputs(part_data)
            if args.fake_data_from:
                indiv_sys_outputs.extend(read_system_training_data(args.fake_data_from))
            fake_insts = create_fake_data(pd.DataFrame.from_records(indiv_sys_outputs), part_df.columns, score_type='rank')
            fake_pairs = create_fake_pairs(fake_insts, len(indiv_sys_outputs))
            part_df = part_df.append(fake_pairs, sort=True)

        out_file = os.path.join(args.out_path, part_label + '.tsv')
        log_info('File: %s, total size %d' % (out_file, len(part_df)))
        part_df.to_csv(out_file, columns=COLUMNS, sep=b"\t", index=False, encoding='UTF-8')

        part_start += part_size


if __name__ == '__main__':
    ap = ArgumentParser(description='Convert E2E data to our format.')
    ap.add_argument('--column', '-d', type=str, help='Column to use (others will be NaN)', required=True)
    ap.add_argument('-u', '--unambiguous', action='store_true',
                    help='Collapse identical MR-hyp-hyp2 tuples & only use the unambiguous ones (via majority vote)')
    ap.add_argument('-s', '--shuffle', action='store_true',
                    help='Shuffle data (within sets)')
    ap.add_argument('-r', '--ratio', type=str, default='8:1:1',
                    help='Train-devel-test split ratio (counted in unique MRs)')
    ap.add_argument('-l', '--labels', type=str, default='train:devel:test',
                    help='Train-devel-test labels (default: train:devel:test)')
    ap.add_argument('-F', '--fake-data-from', type=str,
                    help='Additional file to use for fake data (i.e. system training data)')
    ap.add_argument('-f', '--fake-data', action='store_true',
                    help='Create fake data (by distorting individual system outputs)')

    ap.add_argument('src_file', type=str, help='Path to source data file')
    ap.add_argument('out_path', type=str, help='Output TSV file path (will be concat with <label>.tsv)')

    args = ap.parse_args()
    random.seed(1206)
    convert(args)
