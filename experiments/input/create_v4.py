#!/usr/bin/env python

import errno
import os
import pandas as pd
import sys
import shutil


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


def add_into_train(src1, src2, dst):
    data = [pd.read_csv(os.path.join(src1, 'train.tsv'), sep='\t', encoding='UTF-8', index_col=None)]
    for part in ['train.tsv', 'devel.tsv', 'test.tsv']:
        data.append(pd.read_csv(os.path.join(src2, part), sep='\t', encoding='UTF-8', index_col=None))
    data = pd.concat(data, sort=True).sample(frac=1).reset_index(drop=True)  # concat & shuffle
    data.to_csv(os.path.join(dst, 'train.tsv'), columns=data.columns, sep=b"\t", index=False, encoding='UTF-8')
    shutil.copyfile(os.path.join(src1, 'devel.tsv'), os.path.join(dst, 'devel.tsv'))
    shutil.copyfile(os.path.join(src1, 'test.tsv'), os.path.join(dst, 'test.tsv'))


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


## make directories
#mkdir_p('../data/v4/e2e_qua')
#mkdir_p('../data/v4/e2e_nat')
#mkdir_p('../data/v4/e2e_both')
#mkdir_p('../data/v4/e2e_qua_unamb')
#mkdir_p('../data/v4/e2e_nat_unamb')
#mkdir_p('../data/v4/e2e_both_unamb')
mkdir_p('../data/v4.1/e2e_qua_rt')
#mkdir_p('../data/v4.1/e2e_nat_rt')
#mkdir_p('../data/v4.1/e2e_both_rt')
mkdir_p('../data/v4.1/e2e_qua_unamb_rt')
#mkdir_p('../data/v4.1/e2e_nat_unamb_rt')
#mkdir_p('../data/v4.1/e2e_both_unamb_rt')

#mkdir_p('../data/v4.1/e2e_qua+P')
#mkdir_p('../data/v4.1/e2e_nat+P')
#mkdir_p('../data/v4.1/e2e_both+P')
#mkdir_p('../data/v4.1/e2e_qua_unamb+P')
#mkdir_p('../data/v4.1/e2e_nat_unamb+P')
#mkdir_p('../data/v4.1/e2e_both_unamb+P')
mkdir_p('../data/v4.1/e2e_qua_rt+P')
#mkdir_p('../data/v4.1/e2e_nat_rt+P')
#mkdir_p('../data/v4.1/e2e_both_rt+P')
mkdir_p('../data/v4.1/e2e_qua_unamb_rt+P')
#mkdir_p('../data/v4.1/e2e_nat_unamb_rt+P')
#mkdir_p('../data/v4.1/e2e_both_unamb_rt+P')

## basic data conversion
#os.system('./convert_e2e.py --shuffle --column quality nlg-datasets/quality-fixed_utf.csv ../data/v4/e2e_qua/')
#os.system('./convert_e2e.py --shuffle --fake-data --column quality nlg-datasets/quality-fixed_utf.csv ../data/v4.1/e2e_qua+P')
#os.system('./convert_e2e.py --shuffle --column natur nlg-datasets/naturalness-fixed_utf.csv ../data/v4/e2e_nat/')
#os.system('./convert_e2e.py --shuffle --fake-data --column natur nlg-datasets/naturalness-fixed_utf.csv ../data/v4.1/e2e_nat+P')
#os.system('./convert_e2e.py --unambiguous --shuffle --column quality nlg-datasets/quality-fixed_utf.csv ../data/v4/e2e_qua_unamb/')
#os.system('./convert_e2e.py --unambiguous --shuffle --fake-data --column quality nlg-datasets/quality-fixed_utf.csv ../data/v4.1/e2e_qua_unamb+P')
#os.system('./convert_e2e.py --unambiguous --shuffle --column natur nlg-datasets/naturalness-fixed_utf.csv ../data/v4/e2e_nat_unamb/')
#os.system('./convert_e2e.py --unambiguous --shuffle --fake-data --column natur nlg-datasets/naturalness-fixed_utf.csv ../data/v4.1/e2e_nat_unamb+P')

# ratings in training data
add_into_train('../data/v4/e2e_qua', '../data/v3cv/noref/cv00', '../data/v4.1/e2e_qua_rt')
add_into_train('../data/v4.1/e2e_qua+P', '../data/v3cv/noref/cv00', '../data/v4.1/e2e_qua_rt+P')
add_into_train('../data/v4/e2e_qua_unamb', '../data/v3cv/noref/cv00', '../data/v4.1/e2e_qua_unamb_rt')
add_into_train('../data/v4.1/e2e_qua_unamb+P', '../data/v3cv/noref/cv00', '../data/v4.1/e2e_qua_unamb_rt+P')
#add_into_train('../data/v4/e2e_nat', '../data/v3cv/noref/cv00', '../data/v4.1/e2e_nat_rt')
#add_into_train('../data/v4/e2e_nat+P', '../data/v3cv/noref/cv00', '../data/v4.1/e2e_nat_rt+P')
#add_into_train('../data/v4/e2e_nat_unamb', '../data/v3cv/noref/cv00', '../data/v4.1/e2e_nat_unamb_rt')
#add_into_train('../data/v4/e2e_nat_unamb+P', '../data/v3cv/noref/cv00', '../data/v4.1/e2e_nat_unamb_rt+P')

# concat version
#concat_sets('../data/v4/e2e_qua/', '../data/v4/e2e_nat/', '../data/v4/e2e_both/')
#concat_sets('../data/v4/e2e_qua_unamb/', '../data/v4/e2e_nat_unamb/', '../data/v4/e2e_both_unamb/')
#concat_sets('../data/v4.1/e2e_qua_unamb+P/', '../data/v4.1/e2e_nat_unamb+P/', '../data/v4.1/e2e_both_unamb+P/')

## with ratings data
#for cvnum in ['cv00', 'cv01', 'cv02', 'cv03', 'cv04']:
##    mkdir_p('../data/v4cv/joint/%s' % cvnum)
##    concat_sets('../data/v3cv/noref/%s' % cvnum, '../data/v4/e2e_both/', '../data/v4cv/joint/%s' % cvnum)
    #mkdir_p('../data/v4cv/joint_unamb/%s' % cvnum)
    #concat_sets('../data/v3cv/noref/%s' % cvnum, '../data/v4/e2e_both_unamb/', '../data/v4cv/joint_unamb/%s' % cvnum)

##    mkdir_p('../data/v4cv/joint_small/%s' % cvnum)
##    concat_sets('../data/v3cv/noref/%s' % cvnum, '../data/v4/e2e_both/', '../data/v4cv/joint_small/%s' % cvnum, shorten=5000)
    #mkdir_p('../data/v4cv/joint_small_unamb/%s' % cvnum)
    #concat_sets('../data/v3cv/noref/%s' % cvnum, '../data/v4/e2e_both_unamb/', '../data/v4cv/joint_small_unamb/%s' % cvnum, shorten=5000)

##    mkdir_p('../data/v4cv/joint_Ftonly/%s' % cvnum)
##    concat_sets('../data/v3cv/noref_Ftonly/%s' % cvnum, '../data/v4/e2e_both/', '../data/v4cv/joint_Ftonly/%s' % cvnum)
    #mkdir_p('../data/v4cv/joint_Ftonly_unamb/%s' % cvnum)
    #concat_sets('../data/v3cv/noref_Ftonly/%s' % cvnum, '../data/v4/e2e_both_unamb/', '../data/v4cv/joint_Ftonly_unamb/%s' % cvnum)

## just a test sample for ratings data
#mkdir_p('../data/v4/joint_test')
#concat_sets('../data/v3cv/noref/cv00', '../data/v4/e2e_both/', '../data/v4/joint_test', shorten=500)
