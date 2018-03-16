#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

config = {
          'language': 'en',
          'selector': '',
          'passes': 1000,
          'min_passes': 0,
          'pretrain_passes': 0,
          'randomize': True,
          'batch_size': 5,
          'dropout_keep_prob': 1.0,
          'alpha': 1e-3,
          'emb_size': 10,
          'max_sent_len': 5,
          'max_da_len': 1,
          'max_cores': 4,
          'disk_store_freq': 1,
          'disk_store_min_pass': 20,
          'target_col': 'quality',
          'reuse_embeddings': False,
          'char_embs': False,
          'tanh_layers': 1,
          'predict_ints': False,
          'predict_halves': False,
          'predict_coarse': None, #'train',
          'cell_type': 'gru',
          'bidi': False,
          'ref_enc': False,
          'da_enc': True,
          'use_seq2seq': False,
          'seq2seq_pretrain_passes': 0,
          'seq2seq_min_passes': 0,
          'daclassif_pretrain_passes': 0,
          'daclassif_min_passes': 0,
        }
