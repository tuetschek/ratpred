#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals

from argparse import ArgumentParser
import codecs
import yaml
import re

def main(args):

    iters, training_set, run_setting, nn_shape = '', '', '', ''

    with codecs.open(args.config_file, 'r', 'UTF-8') as fh:
        cfg = yaml.load(fh)

    iters = str(cfg.get('passes', '~'))
    if cfg.get('pretrain_passes'):
        iters = str(cfg['pretrain_passes']) + '^' + iters
    if cfg.get('use_seq2seq'):
        iters = 'S' + str(cfg.get('seq2seq_pretrain_passes', 0)) + ' ' + iters
    if cfg.get('daclassif_pretrain_passes'):
        iters = 'D' + str(cfg.get('daclassif_pretrain_passes')) + ' ' + iters
    iters += '/' + str(cfg.get('batch_size', '~'))
    iters += '/' + (('%.e' % cfg['alpha']).replace('e-0', 'e-') if 'alpha' in cfg else '~')
    if cfg.get('alpha_decay'):
        iters += '^' + str(cfg['alpha_decay'])

    if cfg.get('validation_size'):
        iters = str(cfg.get('min_passes', 1)) + '-' + iters
        iters += ' V' + str(cfg['validation_size'])
        iters += '@' + str(cfg.get('validation_freq', 10))
        iters += ' I' + str(cfg.get('improve_interval', 10))
        iters += '@' + str(cfg.get('top_k', 5))

        if 'validation_weights' in cfg:
            iters += ' ' + ','.join(sorted([
                re.sub('([a-z])[a-z]+(?:_|$)', r'\1', quant.replace('_avg', '')) + str(weight).replace('.0', '')
                for quant, weight in cfg['validation_weights'].iteritems()
            ]))

    training_set = args.training_set
    if args.train_portion < 1.0:
        training_set += '/' + str(args.training_portion)
    training_set += ' -slotn' if cfg.get('delex_slot_names') else ''
    training_set += ' -dlxda' if cfg.get('delex_das') else ''
    training_set += ' +lex' if not cfg.get('delex_slots') else ''

    target_col = 'Q'  # quality is the default
    if 'target_col' in cfg:
        if isinstance(cfg['target_col'], list):
            target_col = ''.join([colname[0].upper() for colname in cfg['target_col']])
        else:
            target_col = cfg['target_col'][0].upper()
    training_set += ' -> ' + target_col

    # gadgets
    nn_shape += ' E' + str(cfg.get('emb_size', 50))
    nn_shape += '-T' + str(cfg.get('tanh_layers', 0))
    if 'dropout_keep_prob' in cfg:
        nn_shape += '-D' + str(cfg['dropout_keep_prob'])

    nn_shape += ' ' + cfg.get('cell_type', 'lstm')
    nn_shape += ' +bidi' if cfg.get('bidi') else ''
    nn_shape += ' +w2v-s' if cfg.get('word2vec_embs') not in [None, 'trainable'] else ''
    nn_shape += ' +w2v-t' if cfg.get('word2vec_embs') == 'trainable' else ''
    nn_shape += ' +ce' if cfg.get('char_embs') else ''
    nn_shape += ' +reuse' if cfg.get('reuse_embeddings') else ''
    nn_shape += ' +da' if cfg.get('da_enc') else ''
    nn_shape += ' -ref' if not cfg.get('ref_enc', True) else ''
    nn_shape += ' -hyp' if not cfg.get('hyp_enc', True) else ''
    nn_shape += ' +1/2s' if cfg.get('predict_halves') else ''
    nn_shape += ' +co-t' if cfg.get('predict_coarse') == 'train' else ''
    nn_shape += ' +co-e' if cfg.get('predict_coarse') == 'test' else ''
    nn_shape += ' +ints' if cfg.get('predict_ints') else ''
    nn_shape += ' +adgr' if cfg.get('optimizer_type') == 'adagrad' else ''
    nn_shape += ' +sqrl' if cfg.get('rank_loss_type') == 'squared' else ''

    if args.cv_runs:
        num_runs = len(args.cv_runs.split())
        run_setting += ' ' + str(num_runs) + 'CV'
    if args.debug:
        run_setting += ' DEBUG'
    if args.rands:
        run_setting += ' RANDS'

    run_setting = run_setting.strip()
    run_setting = run_setting.replace(' ', ',')
    run_setting = ' (' + run_setting + ')' if run_setting else ''

    print(training_set + ' ' + iters + nn_shape + run_setting, end='')


if __name__ == '__main__':

    ap = ArgumentParser()
    ap.add_argument('-t', '--training-set', '--training', type=str, help='Training set name')
    ap.add_argument('-d', '--debug', action='store_true', help='Are we running with debug prints?')
    ap.add_argument('-c', '--cv-runs', '--cv', type=str, help='Number of CV runs used')
    ap.add_argument('-r', '--rands', action='store_true', help='Are we using more random inits?')
    ap.add_argument('-p', '--train-portion', '--portion', type=float, help='Training data portion used', default=1.0)
    ap.add_argument('-e', '--eval-data', '--eval', action='store_true', help='Using evaluation data')
    ap.add_argument('config_file', type=str, help='Experiment YAML config file')

    args = ap.parse_args()
    main(args)
