#!/usr/bin/env python
# -"- coding: utf-8 -"-


import os
from itertools import product

# SETTINGS
settings = [[('-D', 'noref'),
             #('-c', 'concref')
            ],
            [('', ''),
             ('-d dataset=SFHOT', 'xdom'),
             ('-d system=LOLS', 'xsys'),
             ('-d dataset=SFHOT -t -T 200', 'xdomT'),
             ('-d system=LOLS -t -T 200', 'xsysT'),
             ('-d dataset=SFHOT -a 100 -r 0:100:675', 'xdomA'),
             ('-d system=LOLS -a 100 -r 0:100:981', 'xsysA'),
             ('-d dataset=SFHOT -a 100 -r 0:100:675 -R', 'xdomO'),
             ('-d system=LOLS -a 100 -r 0:100:981 -R', 'xsysO'),
            ],
            [('', ''),
             ('-f H', 'fh'),
             ('-f S', 'fs'),
             ('-f TS', 'fts'),
             ('-f HS', 'fhs'),
             ('-f TS -F nlg-datasets/sfxhotel+sfxrest-train.tsv', 'Ftonly'),
             ('-f HS -F nlg-datasets/sfxhotel+sfxrest-train.tsv', 'Ftrain'),
             ('-f HS -F nlg-datasets/bagel+sfxhotel+sfxrest-all.tsv', 'Fall')
            ]]

# plain
print '\nPlain runs\n------'
for setting in product(*settings):
    cmd = ' '.join([flag[0] for flag in setting])
    label = '_'.join([flag[1] for flag in setting]).replace('__', '_').strip('_')
    if not os.path.isdir('../data/v3.2/%s' % label):
        print label
        #print './convert.py -m -s %s nlg-datasets/outputs-refs-scores-indiv_rats.csv ../data/v3.2/%s' % (cmd, label)
        status = os.system('./convert.py -m -s %s nlg-datasets/outputs-refs-scores-indiv_rats.csv ../data/v3.2/%s' % (cmd, label))
        if status:
            exit(status)

## CV (only on full set)
print '\nCV\n------'
for setting in product(settings[0],settings[2]):
    cmd = ' '.join([flag[0] for flag in setting])
    label = '_'.join([flag[1] for flag in setting]).replace('__', '_').strip('_')
    if not os.path.isdir('../data/v3cv/%s' % label):
        print label
        status = os.system('./convert.py -v -m -s %s nlg-datasets/outputs-refs-scores-indiv_rats.csv ../data/v3cv/%s' % (cmd, label))
        if status:
            exit(status)

