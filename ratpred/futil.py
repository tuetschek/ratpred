#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TODO
"""

from __future__ import unicode_literals
import pandas as pd

from tgen.data import DA
from tgen.futil import tokenize
from tgen.delex import delex_sent


def read_data(filename, target_col, delex_slots):
    data = pd.read_csv(filename, sep="\t", encoding='UTF-8')

    das = [DA.parse_cambridge_da(da) for da in data['mr']]
    texts_ref = [[(tok, None)
                  for tok in delex_sent(da, tokenize(sent.lower()).split(' '),
                                        delex_slots, True)[0]]
                 for da, sent in zip(das, data['original_ref_not_modified'])]
    texts_hyp = [[(tok, None)
                  for tok in delex_sent(da, tokenize(sent.lower()).split(' '),
                                        delex_slots, True)[0]]
                 for da, sent in zip(das, data['system_ref'])]

    inputs = [(da, ref, hyp) for da, ref, hyp in zip(das, texts_ref, texts_hyp)]
    targets = data[target_col]

    return inputs, targets

