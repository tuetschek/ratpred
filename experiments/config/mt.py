#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

config = {
          'language': 'en',
          'selector': '',
          'passes': 500,
          'min_passes': 0,
          'randomize': True,
          'batch_size': 100,
          'dropout_keep_prob': 0.5,
          'alpha': 1e-3,
          'emb_size': 100,
          'max_sent_len': 50,
          'validation_size': 100,
          'validation_freq': 1,
          'target_col': 'quality',
          #'target_col': 'naturalness',
          #'target_col': 'informativeness',
          'delex_slot_names': False,
          'reuse_embeddings': True,
          'char_embs': False,
          #'word2vec_embs': 'trainable',
          #'word2vec_model': 'data/word2vec/GoogleNews-vectors-negative-300_100k.bin',
          'tanh_layers': 0,
          'predict_ints': False,
          'predict_halves': True,
          'predict_coarse': None, #'train',
          'cell_type': 'lstm',
          'ref_enc': False,
          'da_enc': True,
        }
