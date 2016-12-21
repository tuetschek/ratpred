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
          'delex_slots': 'name,postcode,address,near,area,phone,addr,count',
          'reuse_embeddings': False,
          'tanh_layers': 0,
          'predict_ints': False,
        }
