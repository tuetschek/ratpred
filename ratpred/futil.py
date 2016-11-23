#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TODO
"""

from __future__ import unicode_literals
import pandas as pd
from tgen.futil import tokenize

def read_data(filename, target_col):
    data = pd.read_csv(filename, sep="\t", encoding='UTF-8')
    texts_ref = [[(tok, None) for tok in tokenize(sent).split(' ')]
                 for sent in data['original_ref_not_modified']]
    texts_hyp = [[(tok, None) for tok in tokenize(sent).split(' ')]
                 for sent in data['system_ref']]
    inputs = [(ref, hyp) for ref, hyp in zip(texts_ref, texts_hyp)]
    targets = data[target_col]
    return inputs, targets




