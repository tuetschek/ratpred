#!/usr/bin/env python
# -"- coding: utf-8 -"-

from argparse import ArgumentParser
from glob import glob
import re

def parse_res(run):
    fname = glob('runs/%s*/test.log.txt' % run)[0]
    #fname = run
    with open(fname, 'r') as fh:
            data = fh.read().replace('\n', ' ')
    pear = re.search(r'Pearson correlation: (.....)', data).group(1)
    spea = re.search(r'Spearman correlation: (.....)', data).group(1)
    mae = re.search(r'MAE: (.....)', data).group(1)
    rmse = re.search(r'RMSE: (.....)', data).group(1)
    return pear, spea, mae, rmse

def main(args):
    runs_buf = []
    for num, run in enumerate(args.runs):
        runs_buf.append(run)
        pear, spea, mae, rmse = parse_res(run)
        print '&',
        print ' & '.join((pear, spea, mae, rmse)),
        if num % args.width == args.width - 1:
            print '\\\\ %% %s' % (' '.join(runs_buf))
            runs_buf = []

if __name__ == '__main__':
    ap = ArgumentParser(description='Grab scores from logs to build LaTeX tables')
    ap.add_argument('--width', '-w', default=1, type=int, help='Table width')
    ap.add_argument('runs', nargs='+', help='Experiment numbers')
    main(ap.parse_args())
