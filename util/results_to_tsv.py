#!/usr/bin/env python2
# -"- coding: utf-8 -"-

from argparse import ArgumentParser
import pandas as pd
import yaml
import os.path
import logging
import codecs
import re


logging.basicConfig(format='%(asctime)-15s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


SCORE_PATTERNS = {
    'dataset': r'  (v[34](?:.[0-9])?(?:cv)?/[a-z_]+) ',
    'exp_number': r'^([0-9]+) ',
    'datetime': r'([0-9]{2}-[0-9]{2}-[0-9]{2} [0-9]{2}-[0-9]{2}-[0-9]{2})',
    'MAE': r': M .\[[0-9;]+m([0-9.]+(?:.\[0m:.\[[0-9;]+m[0-9.]+)*)',
    'RMSE': r'  R .\[[0-9;]+m([0-9.]+(?:.\[0m:.\[[0-9;]+m[0-9.]+)*)',
    'pearson': r'  P .\[[0-9;]+m([0-9.]+(?:.\[0m:.\[[0-9;]+m[0-9.]+)*)',
    'spearman': r'  S .\[[0-9;]+m([0-9.]+(?:.\[0m:.\[[0-9;]+m[0-9.]+)*)',
    'rank_acc': r'  Ar.\[[0-9;]+m([0-9.]+(?:.\[0m:.\[[0-9;]+m[0-9.]+)*)',
    'rank_loss': r'  Lr.\[[0-9;]+m([0-9.]+(?:.\[0m:.\[[0-9;]+m[0-9.]+)*)',
    'options': r'\(([^)]+)\)',
}
SCORE_TYPES = ['MAE', 'RMSE', 'pearson', 'spearman', 'rank_acc', 'rank_loss']
LEAD_COLUMNS = ['exp_number', 'datetime', 'Q_MAE', 'Q_RMSE',
                'Q_pearson', 'Q_spearman', 'Q_rank_acc', 'Q_rank_loss',
                'N_MAE', 'N_RMSE',
                'N_pearson', 'N_spearman', 'N_rank_acc', 'N_rank_loss',
                'dataset', 'options']


def main(exp_dirs, output_tsv):
    data = []
    for exp_dir in exp_dirs:
        if (not os.path.isfile(os.path.join(exp_dir, 'SCORE')) or
                not os.path.isfile(os.path.join(exp_dir, 'config.yaml'))):
            logger.warn('Did not find need files in %s' % exp_dir)
            continue
        # read the necessary files
        with codecs.open(os.path.join(exp_dir, 'SCORE'), 'r', 'UTF-8') as fh:
            score_info = fh.read()
        with codecs.open(os.path.join(exp_dir, 'config.yaml'), 'r', 'UTF-8') as fh:
            cfg = yaml.load(fh)

        cfg['target_col'] = cfg.get('target_col', 'quality')  # default to quality

        # add stuff from the SCORE file
        for key, pat in SCORE_PATTERNS.items():
            m = re.search(pat, score_info)
            if m:
                scores = re.sub('.\[[0-9;]+m', '', m.group(1)).split(':')
                if key in SCORE_TYPES:  # split quality/naturalness scores, prepend with Q_ or N_
                    for score, sc_type in zip(scores, cfg['target_col']):
                        cfg[sc_type[0].upper() + '_' + key] = score
                else:
                    cfg[key] = m.group(1)
            else:
                logger.warn('%s didn\'t match %s' % (exp_dir, key))

        # reformat structured values
        if 'validation_weights' in cfg:
            cfg['validation_weights'] = ','.join(sorted([
                re.sub('([a-z])[a-z]+(?:_|$)', r'\1', quant.replace('_avg', '')) + str(weight).replace('.0', '')
                for quant, weight in cfg['validation_weights'].iteritems()
            ]))
        if 'delex_slots' in cfg:
            cfg['delex_slots'] = ','.join(sorted(cfg['delex_slots'].keys()))
        if isinstance(cfg['target_col'], list):
            cfg['target_col'] = ','.join(cfg['target_col'])

        data.append(cfg)

    # build a TSV file from the result
    df = pd.DataFrame.from_records(data)
    seen = set()
    columns = LEAD_COLUMNS + list(df.columns)
    columns = [col for col in columns if not (col in seen or seen.add(col))]
    df.to_csv(output_tsv, sep='\t', index=False, columns=columns)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('exp_dirs', nargs='+', type=str, help='Experiment directories')
    ap.add_argument('output_tsv', type=str, help='Output TSV path')
    args = ap.parse_args()

    main(args.exp_dirs, args.output_tsv)
