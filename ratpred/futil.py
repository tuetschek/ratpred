#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TODO
"""

from __future__ import unicode_literals
import pandas as pd
import numpy as np

from tgen.data import DA
from tgen.futil import tokenize
from tgen.delex import delex_sent


def read_data(filename, target_col, das_type='cambridge', delex_slots=set(), delex_slot_names=False):
    data = pd.read_csv(filename, sep=b"\t", encoding='UTF-8')

    # force data type to string if the data set doesn't contain human references
    data['orig_ref'] = data['orig_ref'].apply(lambda x: '' if not isinstance(x, basestring) else x)

    def preprocess_sent(da, sent):
        sent = tokenize(sent.lower()).split(' ')
        if delex_slots:
            return delex_sent(da, sent, delex_slots, not delex_slot_names, delex_slot_names)[0]
        return sent

    if das_type == 'text':
        das = [[(tok, None) for tok in preprocess_data(None, sent)]
                for sent in data['mr']]
    else:
        das = [DA.parse_cambridge_da(da) for da in data['mr']]
    texts_ref = [[(tok, None) for tok in preprocess_sent(da, sent)]
                 for da, sent in zip(das, data['orig_ref'])]
    texts_hyp = [[(tok, None) for tok in preprocess_sent(da, sent)]
                 for da, sent in zip(das, data['system_ref'])]

    inputs = [(da, ref, hyp) for da, ref, hyp in zip(das, texts_ref, texts_hyp)]
    targets = data[target_col]

    return inputs, targets


def write_outputs(filename, inputs, raw_targets, targets, raw_outputs, outputs):
    das = [da for da, _, _ in inputs]
    input_refs = [" ".join([tok for tok, _ in input_ref]) for _, input_ref, _ in inputs]
    input_hyps = [" ".join([tok for tok, _ in input_hyp]) for _, _, input_hyp in inputs]
    outputs = [float(output) for output in outputs]
    df = pd.DataFrame({'mr': das,
                       'orig_ref': input_refs,
                       'system_output': input_hyps,
                       'human_rating_raw': raw_targets,
                       'human_rating': targets,
                       'system_rating_raw': raw_outputs,
                       'system_rating': outputs})
    df.to_csv(filename, sep=b"\t", index=False, encoding='UTF-8')
