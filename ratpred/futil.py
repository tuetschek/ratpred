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


def read_data(filename, target_cols, das_type='cambridge',
              delex_slots=set(), delex_slot_names=False, delex_das=False):
    """Read the input data from a TSV file."""

    data = pd.read_csv(filename, sep=b"\t", encoding='UTF-8')

    # force data type to string if the data set doesn't contain human references
    data['orig_ref'] = data['orig_ref'].apply(lambda x: '' if not isinstance(x, basestring) else x)

    def preprocess_sent(da, sent):
        sent = tokenize(sent.lower()).split(' ')
        if delex_slots:
            return delex_sent(da, sent, delex_slots, not delex_slot_names, delex_slot_names)[0]
        return sent

    if das_type == 'text':  # for MT output classification
        das = [[(tok, None) for tok in preprocess_sent(None, sent)]
               for sent in data['mr']]
    else:
        das = [DA.parse_cambridge_da(da) for da in data['mr']]

    texts_ref = [[(tok, None) for tok in preprocess_sent(da, sent)]
                 for da, sent in zip(das, data['orig_ref'])]
    texts_hyp = [[(tok, None) for tok in preprocess_sent(da, sent)]
                 for da, sent in zip(das, data['system_ref'])]

    # DA delexicalization must take place after text delexicalization
    if das_type != 'text' and delex_das:
        das = [da.get_delexicalized(delex_slots) for da in das]

    if 'is_real' in data.columns:
        real_indics = [0 if indic == 0 else 1 for indic in data['is_real']]
    else:
        real_indics = [1 for _ in xrange(len(data))]

    inputs = [(da, ref, hyp, ri)
              for da, ref, hyp, ri in zip(das, texts_ref, texts_hyp, real_indics)]

    targets = np.array(data[[target_cols] if not isinstance(target_cols, list) else target_cols],
                       dtype=np.float)

    return inputs, targets


def write_outputs(filename, inputs, outputs):
    das = [inp[0].to_cambridge_da_string() for inp in inputs]
    input_refs = [" ".join([tok for tok, _ in inp[1]]) for inp in inputs]
    input_hyps = [" ".join([tok for tok, _ in inp[2]]) for inp in inputs]
    df = {'mr': das,
          'orig_ref': input_refs,
          'system_output': input_hyps}
    for target_col in outputs.iterkeys():
        for subcol, values in outputs[target_col].iteritems():
            df[target_col + '_' + subcol] = values
    df = pd.DataFrame(df)
    df.to_csv(filename, sep=b"\t", index=False, encoding='UTF-8')


def read_outputs(filename):
    data = pd.read_csv(filename, sep=b"\t", encoding='UTF-8')
    das = [DA.parse_cambridge_da(da) for da in data['mr']]

    # force string data type for empty human references
    data['orig_ref'] = data['orig_ref'].apply(lambda x: '' if not isinstance(x, basestring) else x)
    texts_ref = [[(tok, None) for tok in tokenize(sent.lower()).split(' ')]
                 for sent in data['orig_ref']]
    texts_hyp = [[(tok, None) for tok in tokenize(sent.lower()).split(' ')]
                 for sent in data['system_output']]
    inputs = [(da, text_ref, text_hyp) for
              da, text_ref, text_hyp in zip(das, texts_ref, texts_hyp)]

    # find out which columns were used for ratings
    target_cols = [c[:-len('_system_rating')] for c in data.columns if c.endswith('_system_rating')]
    assert target_cols
    # compile data from all these columns
    outputs = {}
    for target_col in target_cols:
        outputs[target_col] = {subcol: list(data[target_col + '_' + subcol])
                               for subcol in ['human_rating_raw', 'human_rating',
                                              'system_rating_raw', 'system_rating']}
    return (inputs, outputs)
