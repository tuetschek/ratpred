#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

config = {
          'language': 'en',
          'selector': '',
          'passes': 500,
          'min_passes': 20,
          'randomize': True,
          'batch_size': 20,
          'alpha': 1e-3,
          'emb_size': 50,
          'max_sent_len': 50,
          'validation_size': 100,
          'validation_freq': 1,
          'target_col': 'quality',
          #'target_col': 'naturalness',
          #'target_col': 'informativeness',
          'delex_slots': 'name,postcode,address,near,area,phone,addr,count',
          #'delex_slots': 'count,addr,area,food,price,phone,near,pricerange,postcode,address,eattype,type,price_range,good_for_meal,name',
          'delex_slot_names': False,
          'reuse_embeddings': False,
          'char_embs': False,
          #'word2vec_embs': 'trainable',
          #'word2vec_model': 'data/word2vec/GoogleNews-vectors-negative-300_100k.bin',
          'tanh_layers': 0,
          'predict_ints': True,
          'predict_halves': True,
          'predict_coarse': None, #'train',
          'cell_type': 'lstm',
          'hyp_enc': True,
          'ref_enc': True,
          'da_enc': False,
        }
