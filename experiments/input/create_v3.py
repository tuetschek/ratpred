#!/usr/bin/env python
# -"- coding: utf-8 -"-


import os
from itertools import product

# All configurations to generate
# - Carthesian product is taken
# - tuples: 1st part = convert.py parameters to apply, 2nd part = part of target set name
#
settings = [[('-D', 'noref'),  # not using references when rating (only for data synthesis)
            ],
            [('', ''),
             ('-d dataset=SFHOT -a 100 -r 0:100:675 -R', 'xdomO'), # C1 - domain
             ('-d system=LOLS -a 100 -r 0:100:981 -R', 'xsysO'), # C1 - system
             ('-d dataset=SFHOT -t -T 200', 'xdomT'),  # C2 - domain
             ('-d system=LOLS -t -T 200', 'xsysT'),  # C2 - system
             ('-d dataset=SFHOT -a 100 -r 0:100:675', 'xdomA'),  # C3 - domain
             ('-d system=LOLS -a 100 -r 0:100:981', 'xsysA'), # C3 - system
            ],
            [('', ''), # S1
             ('-f S', 'fs'), # S2
             ('-f S -p only', 'fs-P'), # new
             ('-f S -p add', 'fs+P'), # new
             ('-f TS', 'fts'), # S3
             ('-f TS -p only', 'fts-P'), # new
             ('-f TS -p add', 'fts+P'), # new
             ('-f TS -F nlg-datasets/sfxhotel+sfxrest-train.tsv', 'Ftonly'), # S4
             ('-f TS -F nlg-datasets/sfxhotel+sfxrest-train.tsv -p only', 'Ftonly-P'), # new
             ('-f TS -F nlg-datasets/sfxhotel+sfxrest-train.tsv -p add', 'Ftonly+P'), # new
             ('-f HS -F nlg-datasets/sfxhotel+sfxrest-train.tsv', 'Ftrain'), # S5
             ('-f HS -F nlg-datasets/bagel+sfxhotel+sfxrest-all.tsv', 'Fall'), # S6
             ('-f HS -F nlg-datasets/bagel+sfxhotel+sfxrest-all.tsv -p only', 'Fall-P'), # new
             ('-f HS -F nlg-datasets/bagel+sfxhotel+sfxrest-all.tsv -p add', 'Fall+P'), # new
            ]]

## plain
#print '\nPlain runs\n------'
#for setting in product(*settings):
    #cmd = ' '.join([flag[0] for flag in setting])
    #label = '_'.join([flag[1] for flag in setting]).replace('__', '_').strip('_')
    #if not os.path.isdir('../data/v3.2/%s' % label):
        #print label
        ##print './convert.py -m -s %s nlg-datasets/outputs-refs-scores-indiv_rats.csv ../data/v3.2/%s' % (cmd, label)
        #status = os.system('./convert.py -m -s %s nlg-datasets/outputs-refs-scores-indiv_rats.csv ../data/v3.2/%s' % (cmd, label))
        #if status:
            #exit(status)

## CV (only on full set)
print '\nCV\n------'
for setting in product(settings[0],settings[2]):
    cmd = ' '.join([flag[0] for flag in setting])
    label = '_'.join([flag[1] for flag in setting]).replace('__', '_').strip('_')
    if not os.path.isdir('../data/v3.3cv/%s' % label):
        print label
        print './convert.py -v -m -s %s nlg-datasets/outputs-refs-scores-indiv_rats.csv ../data/v3.3cv/%s' % (cmd, label)
        status = os.system('./convert.py -v -m -s %s nlg-datasets/outputs-refs-scores-indiv_rats.csv ../data/v3.3cv/%s' % (cmd, label))
        if status:
            exit(status)

