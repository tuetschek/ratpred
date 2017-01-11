#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TODO
"""

from __future__ import unicode_literals
import gensim
import numpy as np
import re

from tgen.embeddings import EmbeddingExtract


class Word2VecEmbeddingExtract(EmbeddingExtract):

    def __init__(self, cfg):
        # TODO casing is ignored so far
        if 'word2vec_model' not in cfg:
            raise Exception('Need loaded word2vec model')
        self._w2v = gensim.models.Word2Vec.load_word2vec_format(cfg['word2vec_model'], binary=True)
        self.freq_threshold = cfg.get('emb_freq_threshold', 2)
        self.max_sent_len = cfg.get('max_sent_len', 50)
        self.reverse = cfg.get('reverse', False)

    def init_dict(self, train_sents, dict_ord=None):
        # filter sentences
        occ_counts = {}
        for sent in train_sents:
            for tok, _ in sent:
                occ_counts[tok] = occ_counts.get(tok, 0) + 1
        filt_sents = []
        for sent in train_sents:
            filt_sent = ['<GO>']
            for tok, _ in sent:
                if re.match(r'^[0-9][0-9.]+$', tok):
                    tok = re.sub(r'[0-9]', r'#', tok)
                elif occ_counts[tok] < self.freq_threshold:
                    tok = '<UNK>'
                filt_sent.append(tok)
            filt_sent.append('<STOP>')
            if len(sent) < self.max_sent_len + 2:
                filt_sent.extend(['<VOID>'] * (self.max_sent_len + 2 - len(sent)))
            filt_sent = filt_sent[:self.max_sent_len + 2]
            filt_sents.append(filt_sent)

        self._w2v.min_count = 0
        # this is probably an awful thing to do (initializing the softmax layer by zeroes), but
        # unfortunately they are not stored in the GoogleNews data
        self._w2v.syn1neg = np.zeros((len(self._w2v.vocab), self._w2v.layer1_size), dtype=np.float32)
        self._w2v.build_vocab(filt_sents, keep_raw_vocab=True, update=True)
        self._w2v.train(filt_sents)
        return self.get_w2v_dict_size()

    def get_embeddings(self, sent):
        sent = [re.sub(r'[0-9]', r'#', tok) if re.match(r'^[0-9][0-9.]+$', tok) else tok
                for tok, _ in sent]
        embs = [self._w2v.vocab.get('<GO>').index]
        embs.extend([self._w2v.vocab.get(tok, self._w2v.vocab.get('<UNK>')).index for tok in sent])
        embs.append(self._w2v.vocab.get('<STOP>').index)
        if len(sent) < self.max_sent_len + 2:
            embs.extend([self._w2v.vocab.get('<VOID>').index] * (self.max_sent_len + 2- len(sent)))
        embs = embs[:self.max_sent_len + 2]
        if self.reverse:
            return list(reversed(embs))
        return embs

    def get_embeddings_shape(self):
        """Return the shape of the embedding matrix (for one object, disregarding batches)."""
        return [self.max_sent_len + 2]

    def get_w2v_matrix(self):
        return self._w2v.syn0

    def get_w2v_dict_size(self):
        return len(self._w2v.vocab)
