#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

config = {
          'language': 'en',
          'selector': '',
          'num_hidden_units': 128,
          'passes': 1000,
          'min_passes': 20,
          'randomize': True,
          'batch_size': 20,
          'alpha': 1e-4,
          'emb_size': 50,
          'max_tree_len': 50,
          'validation_freq': 1,
          'target_col': 'quality',
          'delex_slots': 'name,postcode,address,near,area,phone,addr',
          'reuse_embeddings': False,
          'tanh_layers': 0,
        }
