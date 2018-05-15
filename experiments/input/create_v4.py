#!/usr/bin/env python

import errno
import os
import pandas as pd
import sys

def concat_sets(src1, src2, dst, shorten=None):
    for part in ['train.tsv', 'devel.tsv', 'test.tsv']:
        print >> sys.stderr, ('%s + %s -> %s (%s)' % (src1, src2, dst, part))
        data = pd.read_csv(os.path.join(src1, part), sep='\t', encoding='UTF-8', index_col=None)
        add = pd.read_csv(os.path.join(src2, part), sep='\t', encoding='UTF-8', index_col=None)
        if shorten and len(add) > shorten:  # use less data
            add = add.sample(n=shorten)
        data = data.append(add)
        data = data.sample(frac=1).reset_index(drop=True)  # shuffle data
        data.to_csv(os.path.join(dst, part), columns=data.columns, sep=b"\t", index=False, encoding='UTF-8')


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


# make directories
mkdir_p('../data/v4/e2e_qua')
mkdir_p('../data/v4/e2e_nat')
mkdir_p('../data/v4/e2e_both')

# basic data conversion
os.system('./convert_e2e.py --shuffle --column quality nlg-datasets/quality-fixed_utf.csv ../data/v4/e2e_qua/')
os.system('./convert_e2e.py --shuffle --column natur nlg-datasets/naturalness-fixed_utf.csv ../data/v4/e2e_nat/')

# concat version
concat_sets('../data/v4/e2e_qua/', '../data/v4/e2e_nat/', '../data/v4/e2e_both/')

# with ratings data
for cvnum in ['cv00', 'cv01', 'cv02', 'cv03', 'cv04']:
    mkdir_p('../data/v4cv/joint/%s' % cvnum)
    concat_sets('../data/v3cv/noref/%s' % cvnum, '../data/v4/e2e_both/', '../data/v4cv/joint/%s' % cvnum)

    mkdir_p('../data/v4cv/joint_small/%s' % cvnum)
    concat_sets('../data/v3cv/noref/%s' % cvnum, '../data/v4/e2e_both/', '../data/v4cv/joint_small/%s' % cvnum, shorten=5000)

    mkdir_p('../data/v4cv/joint_Ftonly/%s' % cvnum)
    concat_sets('../data/v3cv/noref_Ftonly/%s' % cvnum, '../data/v4/e2e_both/', '../data/v4cv/joint_Ftonly/%s' % cvnum)

# just a test sample for ratings data
mkdir_p('../data/v4/joint_test')
concat_sets('../data/v3cv/noref/cv00', '../data/v4/e2e_both/', '../data/v4/joint_test', shorten=500)
