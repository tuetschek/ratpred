#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File I/O utilities.
"""

from __future__ import unicode_literals
import pandas as pd
import numpy as np

from tgen.data import DA
from tgen.futil import tokenize
from tgen.delex import delex_sent


def preprocess_sent(da, sent, delex_slots, delex_slot_names):
    sent = tokenize(sent.lower()).split(' ')
    if delex_slots:
        return delex_sent(da, sent, delex_slots, not delex_slot_names, delex_slot_names)[0]
    return sent


def interactive_input(das_type='cambridge', delex_slots=set(),
                      delex_slot_names=False, delex_das=False,
                      input_da=True, input_ref=False):

    da = None
    if input_da:
        da = raw_input('Enter DA             : ').decode('utf-8').strip()
        if not da:
            return None
        if das_type == 'text':
            da = [(tok, None) for tok in preprocess_sent(None, da, False, False)]
        else:
            da = DA.parse_cambridge_da(da)
            if delex_das:
                da = da.get_delexicalized(delex_slots)
    ref = None
    if input_ref:
        ref = raw_input('Enter reference      : ').decode('utf-8').strip()
        if not ref:
            return None
        ref = [(tok, None) for tok in preprocess_sent(da, ref, delex_slots, delex_slot_names)]

    hyp = raw_input('Enter system output 1: ').decode('utf-8').strip()
    if not hyp:
        return None
    hyp = [(tok, None) for tok in preprocess_sent(da, hyp, delex_slots, delex_slot_names)]

    hyp2 = raw_input('Enter system output 2: ').decode('utf-8').strip()
    if not hyp2:
        hyp2 = []
    else:
        hyp2 = [(tok, None) for tok in preprocess_sent(da, hyp2, delex_slots, delex_slot_names)]

    return (da, ref, hyp, hyp2)


def read_data(filename, target_cols, das_type='cambridge',
              delex_slots=set(), delex_slot_names=False, delex_das=False):
    """Read the input data from a TSV file."""

    data = pd.read_csv(filename, sep=b"\t", encoding='UTF-8')

    # force data type to string if the data set doesn't contain human references
    data['orig_ref'] = data['orig_ref'].apply(lambda x: '' if not isinstance(x, basestring) else x)

    if das_type == 'text':  # for MT output classification
        das = [[(tok, None) for tok in preprocess_sent(None, sent, False, False)]
               for sent in data['mr']]
    else:
        das = [DA.parse_cambridge_da(da) for da in data['mr']]

    texts_ref = [[(tok, None) for tok in preprocess_sent(da, sent, delex_slots, delex_slot_names)]
                 for da, sent in zip(das, data['orig_ref'])]
    texts_hyp = [[(tok, None) for tok in preprocess_sent(da, sent, delex_slots, delex_slot_names)]
                 for da, sent in zip(das, data['system_ref'])]

    # alternative reference with rating difference / use to compare
    if 'system_ref2' in data.columns:
        texts_hyp2 = [[(tok, None) for tok in preprocess_sent(da, sent, delex_slots, delex_slot_names)]
                      if isinstance(sent, basestring) else None
                      for da, sent in zip(das, data['system_ref2'])]
    else:
        texts_hyp2 = [None] * len(texts_hyp)

    # DA delexicalization must take place after text delexicalization
    if das_type != 'text' and delex_das:
        das = [da.get_delexicalized(delex_slots) for da in das]

    # fake data indicator
    if 'is_real' in data.columns:
        real_indics = [0 if indic == 0 else 1 for indic in data['is_real']]
    else:
        real_indics = [1 for _ in xrange(len(data))]

    inputs = [(da, ref, hyp, hyp2, ri)
              for da, ref, hyp, hyp2, ri in zip(das, texts_ref, texts_hyp, texts_hyp2, real_indics)]

    targets = np.array(data[[target_cols] if not isinstance(target_cols, list) else target_cols],
                       dtype=np.float)

    return inputs, targets


def write_outputs(filename, inputs, outputs):
    das = [inp[0].to_cambridge_da_string() for inp in inputs]
    input_refs = [" ".join([tok for tok, _ in inp[1]]) for inp in inputs]
    input_hyps = [" ".join([tok for tok, _ in inp[2]]) for inp in inputs]
    input_hyp2s = [" ".join([tok for tok, _ in inp[3]]) if inp[3] is not None else None
                   for inp in inputs]
    df = {'mr': das,
          'orig_ref': input_refs,
          'system_output': input_hyps,
          'system_output2': input_hyp2s}

    for target_col in outputs.iterkeys():
        for subcol, values in outputs[target_col].iteritems():
            df[target_col + '_' + subcol] = values

    df = pd.DataFrame(df)
    df.to_csv(filename, sep=b"\t", index=False, encoding='UTF-8')


def read_outputs(filename):
    data = pd.read_csv(filename, sep=b"\t", encoding='UTF-8')
    if isinstance(data.iloc[len(data) - 1]['mr'], float):
        # XXX workaround to a strange bug that sometimes happens -- not sure how to get rid of it,
        # probably an error in Pandas
        print('!!!Strangely need to remove an empty intstance from the end of %s' % filename)
        data = data[:-1]
    das = [DA.parse_cambridge_da(da) for da in data['mr']]

    # force string data type for empty human references
    data['orig_ref'] = data['orig_ref'].apply(lambda x: '' if not isinstance(x, basestring) else x)
    texts_ref = [[(tok, None) for tok in tokenize(sent.lower()).split(' ')]
                 for sent in data['orig_ref']]
    texts_hyp = [[(tok, None) for tok in tokenize(sent.lower()).split(' ')]
                 for sent in data['system_output']]
    if 'system_output2' not in data:
        data['system_output2'] = [None] * len(data)
    texts_hyp2 = [[(tok, None) for tok in tokenize(sent.lower()).split(' ')]
                  if isinstance(sent, basestring) else None
                  for sent in data['system_output2']]
    inputs = [(da, text_ref, text_hyp, text_hyp2) for
              da, text_ref, text_hyp, text_hyp2 in zip(das, texts_ref, texts_hyp, texts_hyp2)]

    # find out which columns were used for ratings
    target_cols = [c[:-len('_system_rating')] for c in data.columns if c.endswith('_system_rating')]
    assert target_cols
    # compile data from all these columns
    outputs = {}
    for target_col in target_cols:
        outputs[target_col] = {subcol: list(data[target_col + '_' + subcol])
                               for subcol in ['human_rating_raw', 'human_rating',
                                              'system_rating_raw', 'system_rating',
                                              'rank_loss', 'rank_ok']}
    return (inputs, outputs)
