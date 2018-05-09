#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The main NLG QE code (Rating Predictor).
"""

from __future__ import unicode_literals
import cPickle as pickle
import time
import datetime
import sys
import re
import math
import os.path
import functools

import numpy as np
import tensorflow as tf

from pytreex.core.util import file_stream

from tgen.logf import log_info, log_debug
from tgen.rnd import rnd
from tgen.embeddings import TokenEmbeddingSeq2SeqExtract, DAEmbeddingSeq2SeqExtract
from tgen.tf_ml import TFModel
from tgen.features import Features
from tgen.ml import DictVectorizer

import tgen.externals.seq2seq as tf06s2s

from ratpred.futil import read_data
from ratpred.embeddings import Word2VecEmbeddingExtract, CharEmbeddingExtract
from ratpred.tb_logging import DummyTensorBoardLogger, TensorBoardLogger
from ratpred.eval import Evaluator


def sigmoid(nums):
    return 1 / (1 + np.exp(-nums))


class RatingPredictor(TFModel):
    """Main rating prediction (QE) class."""

    def __init__(self, cfg):

        super(RatingPredictor, self).__init__(scope_name='predict_rating-' +
                                              cfg.get('scope_suffix', ''))
        self.cfg = cfg
        self.emb_size = cfg.get('emb_size', 50)
        self.cell_type = cfg.get('cell_type', 'lstm')
        self.bidi = cfg.get('bidi', False)
        cfg['reverse'] = True  # embeddings should always be reversed
        self.word2vec_embs = cfg.get('word2vec_embs', None)
        self.char_embs = cfg.get('char_embs', False)
        if self.word2vec_embs and 'word2vec_model' in cfg:
            self.embs = Word2VecEmbeddingExtract(cfg)
        elif self.char_embs:
            self.embs = CharEmbeddingExtract(cfg)
        else:
            self.embs = TokenEmbeddingSeq2SeqExtract(cfg)

        self.da_enc = cfg.get('da_enc', False)
        if self.da_enc:
            if self.da_enc == 'token':
                self.da_embs = TokenEmbeddingSeq2SeqExtract(cfg)
            self.da_embs = DAEmbeddingSeq2SeqExtract(cfg)

        self.ref_enc = cfg.get('ref_enc', True)
        self.hyp_enc = cfg.get('hyp_enc', True)
        if not (self.ref_enc or self.hyp_enc or self.da_enc):
            raise ValueError('At least one encoder must be present!')

        self.passes = cfg.get('passes', 200)
        self.min_passes = cfg.get('min_passes', 0)
        self.pretrain_passes = cfg.get('pretrain_passes', self.passes)

        self.alpha = cfg.get('alpha', 0.1)
        self.randomize = cfg.get('randomize', True)
        self.batch_size = cfg.get('batch_size', 1)
        self.dropout_keep_prob = cfg.get('dropout_keep_prob', 1.0)

        self.validation_size = cfg.get('validation_size', 0)
        self.validation_freq = cfg.get('validation_freq', 10)
        self.validation_weights = cfg.get('validation_weights', {'accuracy': 1.0,
                                                                 'dist_total': 1.0,
                                                                 'cost_total': 0.01, })
        self.max_cores = cfg.get('max_cores')
        self.disk_store_freq = cfg.get('disk_store_freq', self.passes)
        self.disk_store_min_pass = cfg.get('disk_store_min_pass', self.passes)
        self.checkpoint = None
        self.checkpoint_pass = -1
        self.disk_stored_pass = -1

        self.target_cols = cfg.get('target_col', 'quality')
        if not isinstance(self.target_cols, (list, tuple)):
            self.target_cols = [self.target_cols]
        self.delex_slots = cfg.get('delex_slots', set())
        if self.delex_slots:
            self.delex_slots = set(self.delex_slots.split(','))
        self.delex_slot_names = cfg.get('delex_slot_names', False)
        self.delex_das = cfg.get('delex_das', False)

        self.use_seq2seq = cfg.get('use_seq2seq', False)
        self.seq2seq_pretrain_passes = cfg.get('seq2seq_pretrain_passes', 0)
        self.seq2seq_min_passes = cfg.get('seq2seq_min_passes', 0)

        self.daclassif_pretrain_passes = cfg.get('daclassif_pretrain_passes', 0)
        if self.daclassif_pretrain_passes and self.use_seq2seq:
            raise ValueError('Cannot use seq2seq and DA classification pretraining together!')
        self.daclassif_min_passes = cfg.get('daclassif_min_passes', 0)

        self.reuse_embeddings = cfg.get('reuse_embeddings', False)
        self.tanh_layers = cfg.get('tanh_layers', 0)
        self.predict_ints = cfg.get('predict_ints', False)
        self.predict_halves = cfg.get('predict_halves', False)
        self.predict_coarse = cfg.get('predict_coarse', None)
        self.scale_reversed = cfg.get('scale_reversed', False)  # HTER is reversed
        self.num_outputs = None  # will be changed in training

        self.tb_logger = DummyTensorBoardLogger()

    def set_tensorboard_logging(self, log_dir, run_id):
        self.tb_logger = TensorBoardLogger(log_dir, run_id)

    def save_to_file(self, model_fname):
        """Save the predictor to a file (actually two files, one for configuration and one
        for the TensorFlow graph, which must be stored separately).

        @param model_fname: file name (for the configuration file); TF graph will be stored with a \
            different extension
        @param skip_settings: if True, saving settings is skipped, only the model parameters are \
            written to disk
        """
        log_info("Saving model settings to %s..." % model_fname)
        with file_stream(model_fname, 'wb', encoding=None) as fh:
            pickle.dump(self.get_all_settings(), fh, protocol=pickle.HIGHEST_PROTOCOL)
        tf_session_fname = re.sub(r'(.pickle)?(.gz)?$', '.tfsess', model_fname)
        log_info("Saving model parameters to %s..." % tf_session_fname)
        self.saver.save(self.session, os.path.abspath(tf_session_fname))
        log_info('Done.')

    def get_all_settings(self):
        """Get all settings except the trained model parameters (to be stored in a pickle)."""
        data = {'cfg': self.cfg,
                'embs': self.embs, }
        if self.embs:
            data['dict_size'] = self.dict_size
            data['input_shape'] = self.input_shape
            data['outputs_range_lo'] = self.outputs_range_lo
            data['outputs_range_hi'] = self.outputs_range_hi
            data['num_outputs'] = self.num_outputs
        if self.da_enc:
            data['da_embs'] = self.da_embs
            data['da_dict_size'] = self.da_dict_size
            data['da_input_shape'] = self.da_input_shape
        if self.daclassif_pretrain_passes:
            data['daclassif_vect'] = self.daclassif_vect
            data['daclassif_feats'] = self.daclassif_feats
        return data

    def _save_checkpoint(self, top_cost, iter_no, cost, model_fname=None):
        """If the current cost is better than the top cost, store an in-memory checkpoint
        containing all variables and settings of the model. Will always overwrite the
        last checkpoint. Store an on-disk checkpoint based on current iteration number."""
        # check if we need a new checkpoint
        if math.isnan(top_cost) or cost < top_cost:
            log_info('Storing in-memory checkpoint...')
            self.checkpoint = (self.get_all_settings(), self.get_model_params())
            self.checkpoint_pass = iter_no
            log_info('Done.')

        # once in a while, save current best checkpoint to disk
        if (model_fname and
                (iter_no > self.disk_store_min_pass) and
                (iter_no - self.disk_stored_pass >= self.disk_store_freq) and
                (self.checkpoint_pass > self.disk_stored_pass)):
            log_info('Storing last checkpoint to disk...')
            with file_stream(model_fname, 'wb', encoding=None) as fh:
                settings, params = self.checkpoint
                pickle.dump(settings, fh, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(params, fh, protocol=pickle.HIGHEST_PROTOCOL)
            self.disk_stored_pass = iter_no
            log_info('Done.')

    def _restore_checkpoint(self):
        if not self.checkpoint:
            return
        log_info('Restoring in-memory checkpoint...')
        settings, params = self.checkpoint
        self.load_all_settings(settings)
        self.set_model_params(params)

    @staticmethod
    def load_from_file(model_fname):
        """Load the predictor from a file (actually two files, one for configuration and one
        for the TensorFlow graph, which must be stored separately).

        @param model_fname: file name (for the configuration file); TF graph must be stored with a \
            different extension
        """
        log_info("Loading predictor from %s..." % model_fname)
        with file_stream(model_fname, 'rb', encoding=None) as fh:
            data = pickle.load(fh)
            ret = RatingPredictor(cfg=data['cfg'])
            ret.load_all_settings(data)
            ret._build_neural_network()  # rebuild TF graph
            try:
                # load model params from pickle (fails if they're not there)
                ret.set_model_params(pickle.load(fh))
                return ret
            except:
                pass

        # load model params by TF saver
        log_info("Looking for TF saved session...")
        tf_session_fname = os.path.abspath(re.sub(r'(.pickle)?(.gz)?$', '.tfsess', model_fname))
        ret.saver.restore(ret.session, tf_session_fname)
        return ret

    def load_data(self, data_file):
        """Load a data file, return inputs and targets."""
        return read_data(data_file, self.target_cols,
                         'text' if self.da_enc == 'token' else 'cambridge',
                         self.delex_slots, self.delex_slot_names, self.delex_das)

    def _seq2seq_pretrain(self, model_fname=None):

        log_info('Seq2seq pretraining...')
        log_info('Using %d instances.' % sum(len(batch) for batch in self._seq2seq_train_batches()))

        top_cost = float('nan')

        for pass_no in xrange(1, self.seq2seq_pretrain_passes + 1):
            if self.randomize:
                rnd.shuffle(self.train_order)

            self._seq2seq_training_pass(pass_no)

            if (self.valid_inputs and self.validation_freq and
                    pass_no > self.seq2seq_min_passes and pass_no % self.validation_freq == 0):

                cost = self._seq2seq_evaluate(self.valid_inputs, self.valid_y)
                self._print_seq2seq_validation_stats(pass_no, cost)

                # if we have the best model so far, save it as a checkpoint (overwrite previous)
                self._save_checkpoint(top_cost, pass_no, cost, model_fname)
                # remember the current top cost
                top_cost = min(top_cost, cost) if not math.isnan(top_cost) else cost

        # restore the best parameters so far
        self._restore_checkpoint()

    def _daclassif_pretrain(self, model_fname=None):
        log_info('DA classification pretraining...')
        log_info('Using %d instances.' % sum(len(batch) for batch in self._seq2seq_train_batches()))

        top_cost = float('nan')

        for pass_no in xrange(1, self.daclassif_pretrain_passes + 1):
            if self.randomize:
                rnd.shuffle(self.train_order)

            self._daclassif_training_pass(pass_no)

            if (self.valid_inputs and self.validation_freq and
                    pass_no > self.daclassif_min_passes and pass_no % self.validation_freq == 0):

                cost = self._daclassif_evaluate(self.valid_inputs, self.valid_y)
                self._print_daclassif_validation_stats(pass_no, cost)

                # if we have the best model so far, save it as a checkpoint (overwrite previous)
                self._save_checkpoint(top_cost, pass_no, cost, model_fname)
                # remember the current top cost
                top_cost = min(top_cost, cost) if not math.isnan(top_cost) else cost

        # restore the best parameters so far
        self._restore_checkpoint()

    def train(self, train_data_file, valid_data_file=None, data_portion=1.0, model_fname=None):
        """Run training on the given training data.
        """
        inputs, targets = self.load_data(train_data_file)
        valid_inputs, valid_targets = None, None
        if valid_data_file:
            valid_inputs, valid_targets = self.load_data(valid_data_file)
        log_info('Training rating predictor...')

        # initialize training
        self._init_training(inputs, targets, valid_inputs, valid_targets, data_portion)

        # pretrain using seq2seq decoding / DA classification
        if self.seq2seq_pretrain_passes > 0:
            self._seq2seq_pretrain(model_fname)
        if self.daclassif_pretrain_passes > 0:
            self._daclassif_pretrain(model_fname)

        # start training
        top_cost = float('nan')

        log_info('Starting passes...')
        for iter_no in xrange(1, self.passes + 1):
            self.train_order = range(len(self.train_hyps))
            if self.randomize:
                rnd.shuffle(self.train_order)

            # the actual training pass
            pass_cost = self._training_pass(iter_no)

            # validation
            if (self.valid_inputs and self.validation_freq and
                    iter_no > self.min_passes and iter_no % self.validation_freq == 0):

                results = self.evaluate(self.valid_inputs, self.valid_y)
                results['cost_total'] = pass_cost
                results['cost_avg'] = pass_cost / np.sum(~np.isnan(self.valid_y))
                self._compute_comb_cost(results)
                self._print_valid_stats(iter_no, results)

                # if we have the best model so far, save it as a checkpoint (overwrite previous)
                self._save_checkpoint(top_cost, iter_no + self.seq2seq_pretrain_passes,
                                      results['cost_comb'], model_fname)
                # remember the current top cost
                top_cost = (min(top_cost, results['cost_comb'])
                            if not math.isnan(top_cost) else
                            results['cost_comb'])

            if iter_no == self.pretrain_passes:  # continue training only on real data
                self._remove_fake_training_data()

        # restore last checkpoint (best performance on devel data)
        self._restore_checkpoint()

    def _remove_fake_training_data(self):
        """Remove 'fake' training data from the list of training instances, based on the
        self.train_is_reals variable."""
        def filter_real(data):
            return [inst for inst, ir in zip(data, self.train_is_reals) if ir]

        if self.hyp_enc:
            self.train_hyps = filter_real(self.train_hyps)
            self.X_hyp = np.array(filter_real(self.X_hyp))
        if self.ref_enc:
            self.train_refs = filter_real(self.train_refs)
            self.X_ref = np.array(filter_real(self.X_ref))
        if self.da_enc:
            self.train_das = filter_real(self.train_das)
            self.X_da = np.array(filter_real(self.X_da))

        self.y = np.array(filter_real(self.y))
        self.train_is_reals = filter_real(self.train_is_reals)  # basically set all to 1
        log_info("Removed fake training data, %d instances remaining." % len(self.y))

    def _compute_comb_cost(self, results):
        """Compute combined cost, given my validation quantity weights."""
        aspects = [aspect for aspect in results.iterkeys() if isinstance(results[aspect], dict)]
        comb_cost = 0.0
        for quantity, weight in self.validation_weights.iteritems():
            if quantity in results:
                comb_cost += weight * results[quantity]
            else:  # sum the given quantity over all target columns equally
                comb_cost += sum([weight * results[aspect][quantity] for aspect in aspects])
        results['cost_comb'] = comb_cost
        return comb_cost

    def rate(self, hyps=None, refs=None, das=None):
        """
        Rate a pair of reference sentence + system output hypothesis.

        @param refs: a reference sentence (as a 1-element array, batches not yet supported)
        @param hyps: a system output hypothesis (as a 1-element array, batches not yet supported)
        @return: the rating, as a floating point number (not rounded to prediction boundaries)
        """
        inputs_hyp = np.array([self.embs.get_embeddings(sent) for sent in hyps]) if hyps else None
        inputs_ref = np.array([self.embs.get_embeddings(sent) for sent in refs]) if refs else None
        inputs_da = np.array([self.da_embs.get_embeddings(da) for da in das]) if das else None
        fd = {}
        self._add_inputs_to_feed_dict(fd, inputs_hyp, inputs_ref, inputs_da)
        val = self.session.run(self.output, feed_dict=fd)
        return self._adjust_output(val).astype(float)

    def _adjust_output(self, val, no_sigmoid=False):
        if self.predict_ints:
            # do the actual sigmoid + squeeze it into our range (using defined coarseness)
            coeff = self._rounding_step('train')
            if not no_sigmoid:
                val = sigmoid(val)
            val = val.reshape(val.shape[:1] + self.num_outputs)
            return np.clip(coeff * np.sum(val, axis=2) + self.outputs_range_lo,
                           self.outputs_range_lo, self.outputs_range_hi)
        else:
            # just squeeze the output float value into our range
            return np.clip(val, self.outputs_range_lo, self.outputs_range_hi)

    def _divide_inputs(self, inputs, trunc_size=None, get_is_real=False):
        """Divide different types of input data (DAs, references, system outputs, real data
        indicators) into separate lists.

        @param inputs: the input data (as one list of tuples)
        @param trunc_size: truncate the data to the given size (do nothing if `None`)
        @param get_is_real: return the `is_real` indicators as well
        @return: the input data, as tuple of lists (three or four-tuple, based on `get_is_real`)
        """
        size = trunc_size if trunc_size is not None else len(inputs)
        das = [inp[0] for inp in inputs[:size]]
        refs = [inp[1] for inp in inputs[:size]]
        hyps = [inp[2] for inp in inputs[:size]]
        hyp2s = [inp[3] for inp in inputs[:size]]
        if get_is_real:
            irs = [inp[4] for inp in inputs[:size]]
            return (das, refs, hyps, hyp2s, irs)
        return (das, refs, hyps, hyp2s)

    def _cut_valid_data(self):
        assert self.validation_size < len(self.train_das)
        self.valid_inputs = (self.train_das[-self.validation_size:],
                             self.train_refs[-self.validation_size:],
                             self.train_hyps[-self.validation_size:],
                             self.train_hyp2s[-self.validation_size:])
        self.valid_y = self.y[-self.validation_size:]
        self.y = self.y[:-self.validation_size]
        self.train_das = self.train_das[:-self.validation_size]
        self.train_refs = self.train_refs[:-self.validation_size]
        self.train_hyps = self.train_hyps[:-self.validation_size]
        self.train_hyp2s = self.train_hyp2s[:-self.validation_size]
        self.train_is_reals = self.train_is_reals[:-self.validation_size]

    def _init_training(self, inputs, targets,
                       valid_inputs=None, valid_targets=None, data_portion=1.0):
        """Initialize training.

        @param data_portion: portion of the training data to be used (0.0-1.0)
        """
        # store training data, make it smaller if necessary
        train_size = int(round(data_portion * len(inputs)))
        self.train_das, self.train_refs, self.train_hyps, self.train_hyp2s, self.train_is_reals = (
            self._divide_inputs(inputs, train_size, get_is_real=True))
        self.y = targets[:train_size]

        self.valid_inputs, self.valid_y = None, None
        if valid_inputs is not None and valid_targets is not None:
            self.valid_inputs = self._divide_inputs(valid_inputs)
            self.valid_y = valid_targets
        elif self.validation_size > 0:
            self._cut_valid_data()

        self.train_order = range(len(self.train_hyps))
        log_info('Using %d training instances.' % len(self.train_hyps))

        # initialize input embedding storage
        if self.hyp_enc:
            self.dict_size = self.embs.init_dict(self.train_hyps)
        else:
            self.dict_size = len(self.embs.dict) + 1
        self.dict_size = self.embs.init_dict([hyp2 for hyp2 in self.train_hyp2s if hyp2 is not None],
                                             dict_ord=self.dict_size)  # ignore null 2nd hyps for initialization
        if self.ref_enc:
            self.dict_size = self.embs.init_dict(self.train_refs, dict_ord=self.dict_size)
        if self.da_enc:
            self.da_dict_size = self.da_embs.init_dict(self.train_das)

        # initialize DAs to binary vectors conversion if pretraining by DA classification
        if self.daclassif_pretrain_passes:
            self.daclassif_feats = Features(['dat: dat_presence', 'svp: svp_presence'])
            self.daclassif_vect = DictVectorizer(sparse=False, binarize_numeric=True)
            self.daclassif_y = [self.daclassif_feats.get_features(None, {'da': da})
                                for da in self.train_das]
            self.daclassif_y = self.daclassif_vect.fit_transform(self.daclassif_y)

        # convert training data to indexes
        if self.hyp_enc:
            self.X_hyp = np.array([self.embs.get_embeddings(sent) for sent in self.train_hyps])
            self.X_hyp2 = np.array([self.embs.get_embeddings(sent) for sent in self.train_hyp2s])
            self.X_has_hyp2 = np.array([sent is not None for sent in self.train_hyp2s], dtype=np.float)
        if self.ref_enc:
            self.X_ref = np.array([self.embs.get_embeddings(sent) for sent in self.train_refs])
        if self.da_enc:
            self.X_da = np.array([self.da_embs.get_embeddings(da) for da in self.train_das])

        # initialize I/O shapes and boundaries
        self.input_shape = self.embs.get_embeddings_shape()
        if self.da_enc:
            self.da_input_shape = self.da_embs.get_embeddings_shape()

        # find the ranges of the targets (this needs to be global for all targets for simplicity)
        self.outputs_range_lo = np.round(np.min(self.y[~np.isnan(self.y)])).astype(np.int)
        self.outputs_range_hi = np.round(np.max(self.y[~np.isnan(self.y)])).astype(np.int)
        if self.predict_ints:  # var. number of binary outputs (based on coarseness & lo-hi)
            self.y = self._ratings_to_binary(self.y)
        else:  # just one real-valued output per aspect (rounded to desired coarseness)
            self.y = self._round_rating(self.y, mode='train').reshape(self.y.shape + (1,))
        self.num_outputs = self.y.shape[1:]

        # initialize NN classifier
        self._build_neural_network()
        # initialize the NN variables
        self.session.run(tf.global_variables_initializer())

    def _rounding_step(self, mode='test'):
        # TODO predict_coarse was designed for lo-mid-hi ??
        if self.predict_coarse == 'train' or self.predict_coarse is not None and mode == 'test':
            return (self.outputs_range_hi - self.outputs_range_lo) / 2.0
        elif self.predict_halves:
            return 0.5
        return 1.0

    def _round_rating(self, rating, mode='test'):
        step = self._rounding_step(mode)
        return self.outputs_range_lo + np.round((rating - self.outputs_range_lo) / step) * step

    def _ratings_to_binary(self, ints):
        """Transform input int/half-int values into list of binaries (1's lower than
        the (half-)int, 0's higher). Takes a 2-D array (instances x measures), returns
        a 3-D array."""
        step = self._rounding_step('train')
        ints = [[[0 if measure < i + (step / 2.0) else 1
                  for i in np.arange(self.outputs_range_lo, self.outputs_range_hi, step)]
                 for measure in val]
                for val in ints]
        return np.array(ints)

    def _build_neural_network(self):
        """Create the neural network for classification"""

        # set TensorFlow random seed
        tf.set_random_seed(rnd.randint(-sys.maxint, sys.maxint))

        self.target = tf.placeholder(tf.float32, [None, np.prod(self.num_outputs)], name='target')

        with tf.variable_scope(self.scope_name):

            self.train_mode = tf.placeholder(tf.bool, [], name='train_mode')

            if self.hyp_enc:
                self.inputs_hyp = [tf.placeholder(tf.int32, [None], name=('enc_inp_hyp-%d' % i))
                                   for i in xrange(self.input_shape[0])]
                self.inputs_hyp2 = [tf.placeholder(tf.int32, [None], name=('enc_inp_hyp2-%d' % i))
                                    for i in xrange(self.input_shape[0])]
            if self.ref_enc:
                self.inputs_ref = [tf.placeholder(tf.int32, [None], name=('enc_inp_ref-%d' % i))
                                   for i in xrange(self.input_shape[0])]
            if self.da_enc:
                self.inputs_da = [tf.placeholder(tf.int32, [None], name=('enc_inp_da-%d' % i))
                                  for i in xrange(self.da_input_shape[0])]

            if self.cell_type.startswith('gru'):
                self.cell = tf.contrib.rnn.GRUCell(self.emb_size)
            else:
                self.cell = tf.contrib.rnn.BasicLSTMCell(self.emb_size)
            if self.cell_type.endswith('/2'):
                self.cell = tf.contrib.rnn.MultiRNNCell([self.cell] * 2)

            # build the output
            if self.use_seq2seq:
                assert (self.da_enc and self.hyp_enc and not self.ref_enc)
                # will also build s2s cost, optimizer, and training function
                self._build_seq2seq_net(self.inputs_da, self.inputs_hyp)
            else:
                # build encoders + final classifier
                self._build_classif_net(self.inputs_hyp if self.hyp_enc else None,
                                        self.inputs_hyp2 if self.hyp_enc else None,
                                        self.inputs_ref if self.ref_enc else None,
                                        self.inputs_da if self.da_enc else None)

            # mask for the cost -- do not learn on unannotated stuff if we don't have annotation
            # for all aspects
            self.aspect_mask = tf.placeholder(tf.float32, shape=self.target.shape, name='aspect_mask')

        # classification cost
        if self.predict_ints:
            # sigmoid cost -- predict a bunch of 1's and 0's (not just one 1)
            self.classif_cost = tf.reduce_mean(tf.reduce_sum(
                tf.multiply(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=self.output, labels=self.target, name='CE'),
                    self.aspect_mask),
                axis=1))
        else:
            # mean square error cost -- predict 1 number
            self.classif_cost = tf.reduce_mean(tf.multiply(tf.square(self.target - self.output),
                                                           self.aspect_mask))

        # pairwise ranking cost
        if self.hyp_enc:
            with tf.variable_scope(self.scope_name):
                self.ranking_mask = tf.placeholder(tf.float32, shape=self.target.shape,
                                                   name='ranking_mask')
                # pairwise hinge loss
                pairwise_hinge = tf.maximum(0.0, 1.0 - self.rank_diff)
                # XXX TODO have the option of pairwise square loss
                self.ranking_cost = tf.reduce_mean(tf.multiply(pairwise_hinge, self.ranking_mask))
        else:
            # ignore any ranking cost if we don't have hyps
            self.ranking_cost = tf.constant(0.0)

        self.cost = self.classif_cost + self.ranking_cost

        self.tb_logger.create_tensor_summaries(self.classif_cost)
        self.tb_logger.create_tensor_summaries(self.ranking_cost)
        self.tb_logger.create_tensor_summaries(self.cost)

        self.optimizer = tf.train.AdamOptimizer(self.alpha)
        self.train_func = self.optimizer.minimize(self.cost)
        self.tensor_summaries = self.tb_logger.get_merged_summaries()

        # initialize session
        session_config = None
        if self.max_cores:
            session_config = tf.ConfigProto(inter_op_parallelism_threads=self.max_cores,
                                            intra_op_parallelism_threads=self.max_cores)
        self.session = tf.Session(config=session_config)
        self.tb_logger.add_graph(self.session.graph)

        # this helps us load/save the model
        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2)

    def _dropout(self, variable):
        """Apply dropout to a given TF variable (will be used in training, not in decoding)."""
        if self.dropout_keep_prob == 1.0:
            return variable
        train_mode_mask = tf.fill(tf.shape(variable)[:1], self.train_mode)
        return tf.where(train_mode_mask,
                        tf.nn.dropout(variable, self.dropout_keep_prob),
                        variable)

    def _build_embs(self, enc_inputs_hyp=None, enc_inputs_hyp2=None,
                    enc_inputs_ref=None, enc_inputs_da=None):
        """Build embedding lookups over the inputs.
        @return: a triple of embedding lookup tensors (hyps, refs, das, some of them may be None)
        """
        enc_in_hyp_emb, enc_in_hyp2_emb, enc_in_ref_emb, enc_in_da_emb = None, None, None, None
        # build embeddings
        with tf.variable_scope('embs'):
            if self.word2vec_embs:
                self.emb_storage = tf.Variable(
                    self.embs.get_w2v_matrix(),
                    trainable=(self.word2vec_embs == 'trainable'),
                    name='emb_storage')
                if self.emb_size != self.embs.get_w2v_width():
                    self.emb_transform = tf.get_variable(
                        'emb_transform',
                        (self.embs.get_w2v_width(), self.emb_size),
                        initializer=tf.random_normal_initializer(stddev=0.1))
            else:
                sqrt3 = math.sqrt(3)
                self.emb_storage = tf.get_variable(
                    'emb_storage',
                    (self.dict_size, self.emb_size),
                    initializer=tf.random_uniform_initializer(-sqrt3, sqrt3))
            self.tb_logger.create_tensor_summaries(self.emb_storage)

            if self.word2vec_embs and self.emb_size != self.embs.get_w2v_width():

                def apply_emb(enc_inp):
                    return self._dropout(
                        tf.matmul(tf.nn.embedding_lookup(self.emb_storage, enc_inp),
                                  self.emb_transform))
            else:

                def apply_emb(enc_inp):
                    return self._dropout(tf.nn.embedding_lookup(self.emb_storage, enc_inp))

        if enc_inputs_hyp is not None:
            with tf.variable_scope('enc_hyp') as scope:
                enc_in_hyp_emb = [apply_emb(enc_inp) for enc_inp in enc_inputs_hyp]
                scope.reuse_variables()
                enc_in_hyp2_emb = [apply_emb(enc_inp) for enc_inp in enc_inputs_hyp2]

        if enc_inputs_ref is not None:
            with self._get_ref_variable_scope():
                enc_in_ref_emb = [apply_emb(enc_inp) for enc_inp in enc_inputs_ref]

        if enc_inputs_da is not None:
            with tf.variable_scope('enc_da'):
                sqrt3 = math.sqrt(3)
                self.da_emb_storage = tf.get_variable(
                    'emb_storage',
                    (self.da_dict_size, self.emb_size),
                    initializer=tf.random_uniform_initializer(-sqrt3, sqrt3))
                self.tb_logger.create_tensor_summaries(self.da_emb_storage)
                enc_in_da_emb = [self._dropout(tf.nn.embedding_lookup(self.da_emb_storage, enc_inp))
                                 for enc_inp in enc_inputs_da]

        return enc_in_hyp_emb, enc_in_hyp2_emb, enc_in_ref_emb, enc_in_da_emb

    def _get_ref_variable_scope(self):
        """Get the correct variable scope for reference encoder; depending on whether the system output and
        the reference encoder variables are shared or not."""
        if self.reuse_embeddings and self.hyp_enc:
            scope = tf.variable_scope('enc_hyp')
            scope.reuse_variables()
        else:
            scope = tf.variable_scope('enc_ref')
        return scope

    def _build_classif_net(self, enc_inputs_hyp=None, enc_inputs_hyp2=None,
                           enc_inputs_ref=None, enc_inputs_da=None):
        """Build the rating prediction RNN structure.
        @return: TensorFlow Output with the prediction
        """
        # create embedding lookups
        enc_in_hyp_emb, enc_in_hyp2_emb, enc_in_ref_emb, enc_in_da_emb = self._build_embs(
            enc_inputs_hyp, enc_inputs_hyp2, enc_inputs_ref, enc_inputs_da)

        # select RNN type (ltr or bidi)
        rnn_func = functools.partial(tf.contrib.rnn.static_rnn,
                                     cell=self.cell, dtype=tf.float32)
        if self.bidi:
            rnn_func = functools.partial(tf.contrib.rnn.static_bidirectional_rnn,
                                         cell_fw=self.cell, cell_bw=self.cell, dtype=tf.float32)

        # apply RNN over embeddings
        enc_state_hyp, enc_state_hyp2, enc_state_ref, enc_state_da = None, None, None, None
        if enc_inputs_hyp is not None:
            with tf.variable_scope('enc_hyp') as scope:
                enc_state_hyp = rnn_func(inputs=enc_in_hyp_emb)[1:]
                scope.reuse_variables()
                enc_state_hyp2 = rnn_func(inputs=enc_in_hyp2_emb)[1:]

        if enc_inputs_ref is not None:
            with self._get_ref_variable_scope():
                enc_state_ref = rnn_func(inputs=enc_in_ref_emb)[1:]

        if enc_inputs_da is not None:
            with tf.variable_scope('enc_da'):
                enc_state_da = rnn_func(inputs=enc_in_da_emb)[1:]

        if self.daclassif_pretrain_passes > 0:
            assert enc_inputs_hyp is not None
            self._build_da_classifier(tf.concat(self._flatten_enc_state(enc_state_hyp), axis=1))

        # concatenate last LSTM states & outputs (works for bidi & multilayer LSTMs&GRUs)
        def concat_enc_states(hyp, ref, da, var_name):
            return tf.concat((self._flatten_enc_state(hyp) +
                              self._flatten_enc_state(ref) +
                              self._flatten_enc_state(da)),
                             axis=1, name=var_name)

        last_outs_and_states = concat_enc_states(enc_state_hyp, enc_state_ref, enc_state_da,
                                                 'last_outs_and_states')
        if enc_inputs_hyp is not None:
            last_outs_and_states2 = concat_enc_states(enc_state_hyp2, enc_state_ref, enc_state_da,
                                                      'last_outs_and_states2')
        else:
            last_outs_and_states2 = None

        # build final FF layers + self.output
        self._build_final_classifier(last_outs_and_states, last_outs_and_states2)

    def _build_da_classifier(self, final_state):

        num_da_classes = len(self.daclassif_vect.get_feature_names())
        log_info('Number of DA classification classes: %d.' % num_da_classes)

        state_size = int(final_state.get_shape()[1])

        with tf.variable_scope('daclassif'):
            # placeholder for target binary vectors
            self.daclassif_targets = tf.placeholder(tf.float32, [None, num_da_classes],
                                                    name='targets')
            # final classification layer
            w = tf.get_variable('daclassif-w', (state_size, num_da_classes),
                                initializer=tf.random_normal_initializer(stddev=0.1))
            b = tf.get_variable('daclassif-b', (num_da_classes,),
                                initializer=tf.constant_initializer())
            self.daclassif_output = tf.matmul(final_state, w) + b

            # training criterion & optimizer
            self.daclassif_cost = tf.reduce_mean(tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(self.daclassif_output,
                                                        self.daclassif_targets, name='CE'), 1))
            self.daclassif_optimizer = tf.train.AdamOptimizer(self.alpha)
            self.daclassif_train_func = self.daclassif_optimizer.minimize(self.daclassif_cost)

    def _build_final_classifier(self, final_state, final_state2):
        """Build the final feedforward layers for the classification."""
        self.tb_logger.create_tensor_summaries(final_state)
        state_size = int(final_state.get_shape()[1])

        hidden = final_state
        hidden2 = final_state2
        if self.tanh_layers > 0:
            with tf.variable_scope('hidden'):
                for layer_no in xrange(self.tanh_layers):
                    h_w = tf.get_variable('final-ff-w' + str(layer_no + 1),
                                          (state_size, state_size),
                                          initializer=tf.random_normal_initializer(stddev=0.1))
                    h_b = tf.get_variable('final-ff-b' + str(layer_no + 1),
                                          (state_size,),
                                          initializer=tf.constant_initializer())
                    hidden = tf.tanh(tf.matmul(hidden, h_w) + h_b)
                    self.tb_logger.create_tensor_summaries(hidden)
                    if final_state2 is not None:
                        hidden2 = tf.tanh(tf.matmul(hidden2, h_w) + h_b)
                        self.tb_logger.create_tensor_summaries(hidden2)

        with tf.variable_scope('classif'):
            # even though the outputs are 2-D, the final layer is flattened for simple
            # matrix multiplication
            w = tf.get_variable('final-transf-w', (state_size, np.prod(self.num_outputs)),
                                initializer=tf.random_normal_initializer(stddev=0.1))
            b = tf.get_variable('final-transf-b', (np.prod(self.num_outputs)),
                                initializer=tf.constant_initializer())
            self.tb_logger.create_tensor_summaries(w)
            self.tb_logger.create_tensor_summaries(b)

            # create output variable
            self.output = tf.matmul(hidden, w) + b
            if final_state2 is not None:
                self.output2 = tf.matmul(hidden2, w) + b
            else:
                self.output2 = tf.constant(0.0, shape=self.output.shape)

            self.rank_diff = self.output - self.output2

    def _build_seq2seq_net(self, enc_inputs, dec_inputs):
        """Build the seq2seq part of the network, over the given DA/system output inputs."""

        # seq2seq targets are just decoder inputs shifted by one + padded
        with tf.variable_scope('seq2seq'):
            self.s2s_targets = [dec_inputs[i + 1] for i in xrange(len(dec_inputs) - 1)]
            self.s2s_targets.append(tf.placeholder(tf.int32, [None], name=('target-pad')))

        # build embedding lookups
        dec_in_emb, _, enc_in_emb = self._build_embs(dec_inputs, None, enc_inputs)

        # build the seq2seq network
        with tf.variable_scope('seq2seq') as scope:
            # encoder
            enc_outputs, enc_states = tf06s2s.rnn(self.cell, enc_in_emb, dtype=tf.float32)

            # a concatenation of encoder outputs to put attention on
            top_states = [tf.reshape(e, [-1, 1, self.cell.output_size]) for e in enc_outputs]
            att_states = tf.concat(1, top_states)

            # decoder
            dec_cell = tf.contrib.rnn.OutputProjectionWrapper(self.cell, self.dict_size)
            dec_outputs, dec_states = tf06s2s.embedding_attention_decoder(
                dec_inputs, enc_states[-1], att_states, dec_cell,
                self.dict_size, self.emb_size, 1, self.dict_size,
                feed_previous=False, scope=scope)

            # decoding cost, optimization function
            s2s_cost_weights = [tf.ones_like(trg, tf.float32, name='cost_weights')
                                for trg in self.s2s_targets]
            self.s2s_cost = tf06s2s.sequence_loss(dec_outputs, self.s2s_targets,
                                                  s2s_cost_weights, self.dict_size)
            self.s2s_optimizer = tf.train.AdamOptimizer(self.alpha)
            self.s2s_train_func = self.s2s_optimizer.minimize(self.s2s_cost)

        # build the final classification layer(s), using last seq2seq output
        final_state = tf.concat(1, self._flatten_enc_state(dec_states[-1]))
        self._build_final_classifier(final_state)

    def _batches(self):
        """Create batches from the input; use as iterator."""
        for i in xrange(0, len(self.train_order), self.batch_size):
            yield self.train_order[i: i + self.batch_size]

    def _flatten_enc_state(self, enc_state):
        """Flatten up to 3 dimensions of tuples, return 1-D array."""
        if enc_state is None:
            return []
        if isinstance(enc_state, tuple):
            if isinstance(enc_state[0], tuple):
                if isinstance(enc_state[0][0], tuple):
                    return [x for z in enc_state for y in z for x in y]
                return [x for y in enc_state for x in y]
            return [x for x in enc_state]
        return [enc_state]

    def _add_inputs_to_feed_dict(self, fd,
                                 inputs_hyp=None, inputs_hyp2=None, inputs_ref=None, inputs_da=None,
                                 inputs_s2s_trg=False, inputs_daclassif_trg=None,
                                 train_mode=False):
        """Add inputs into TF feed_dict."""
        fd[self.train_mode] = train_mode

        if inputs_hyp is not None:
            # TODO check for none when squeezing ?
            sliced_hyp = np.squeeze(np.array(np.split(inputs_hyp, len(inputs_hyp[0]), axis=1)), axis=2)
            for input_, slice_ in zip(self.inputs_hyp, sliced_hyp):
                fd[input_] = slice_
            sliced_hyp2 = np.squeeze(np.array(np.split(inputs_hyp2, len(inputs_hyp2[0]), axis=1)), axis=2)
            for input_, slice_ in zip(self.inputs_hyp2, sliced_hyp2):
                fd[input_] = slice_

            if inputs_s2s_trg:
                sliced_trg = np.concatenate((sliced_hyp[1:],
                                             np.array(len(sliced_hyp[0]) *
                                                      [self.embs.VOID])[np.newaxis]), axis=0)
                for input_, slice_ in zip(self.s2s_targets, sliced_trg):
                    fd[input_] = slice_

        if inputs_ref is not None:
            sliced_ref = np.squeeze(np.array(np.split(inputs_ref, len(inputs_ref[0]), axis=1)), axis=2)
            for input_, slice_ in zip(self.inputs_ref, sliced_ref):
                fd[input_] = slice_

        if inputs_da is not None:
            sliced_da = np.squeeze(np.array(np.split(inputs_da, len(inputs_da[0]), axis=1)), axis=2)
            for input_, slice_ in zip(self.inputs_da, sliced_da):
                fd[input_] = slice_

        if inputs_daclassif_trg is not None:
            fd[self.daclassif_targets] = inputs_daclassif_trg

    def _training_pass(self, pass_no):
        """Perform one training pass through the whole training data, print statistics."""

        pass_start_time = time.time()

        log_debug('\n***\nTR %05d:' % pass_no)
        log_debug("Train order: " + str(self.train_order))

        pass_insts = 0
        pass_cost, pass_classif_cost, pass_ranking_cost = 0.0, 0.0, 0.0
        pass_valid = np.zeros(self.y.shape[1])  # float for easy accuracy computation
        pass_corr = np.zeros(self.y.shape[1], dtype=int)
        pass_dist = np.zeros(self.y.shape[1])
        pass_corr_ranks = np.zeros(self.y.shape[1], dtype=int)
        pass_valid_ranks = np.zeros(self.y.shape[1])  # float for easy accuracy computation

        for inst_nos in self._batches():

            pass_insts += len(inst_nos)
            log_debug('INST-NOS: ' + str(inst_nos))
            log_debug("\n".join(' '.join([tok for tok, _ in self.train_hyps[i]]) + "\n" +
                                ' '.join([tok for tok, _ in self.train_hyp2s[i]]
                                         if self.train_hyp2s[i] is not None else ['<NONE>']) + "\n" +
                                ' '.join([tok for tok, _ in self.train_refs[i]]) + "\n" +
                                unicode(self.train_das[i]) + "\n" +
                                unicode(self.y[i])
                                for i in inst_nos))

            targets = self.y[inst_nos]
            targets = np.reshape(targets, (targets.shape[0], np.prod(targets.shape[1:])))
            aspect_mask = 1. - np.isnan(targets)  # 1 for numbers, 0 for NaNs
            targets = np.nan_to_num(targets, copy=False)

            ranking_mask = np.dot(self.X_has_hyp2[inst_nos].reshape(len(inst_nos), 1),
                                  np.ones((1, self.y.shape[1])))
            ranking_mask *= aspect_mask
            aspect_mask *= 1. - ranking_mask

            fd = {self.target: targets,
                  self.aspect_mask: aspect_mask,
                  self.ranking_mask: ranking_mask}
            self._add_inputs_to_feed_dict(fd,
                                          inputs_hyp=self.X_hyp[inst_nos] if self.hyp_enc else None,
                                          inputs_hyp2=self.X_hyp2[inst_nos] if self.hyp_enc else None,
                                          inputs_ref=self.X_ref[inst_nos] if self.ref_enc else None,
                                          inputs_da=self.X_da[inst_nos] if self.da_enc else None,
                                          train_mode=True)

            required = [self.output, self.rank_diff,
                        self.classif_cost, self.ranking_cost, self.cost,
                        self.train_func]
            if pass_insts == len(self.train_hyps):  # last batch
                required.append(self.tensor_summaries)
            outputs = self.session.run(required, feed_dict=fd)
            results = outputs[0]
            classif_cost, ranking_cost, total_cost = outputs[2:5]
            if pass_insts == len(self.train_hyps):
                self.tb_logger.log(pass_no, outputs[-1])

            # rating evaluation
            valid = np.sum(aspect_mask, axis=0)
            pred = self._adjust_output(results)
            true = self._adjust_output(targets, no_sigmoid=True)
            dist = np.sum(aspect_mask * np.abs(pred - true), axis=0)
            corr = np.sum(aspect_mask.astype(np.int) *
                          (self._round_rating(pred) == self._round_rating(true)), axis=0)

            log_debug('R: ' + str(results))
            log_debug('COST: %f, corr %s/%d, dist %s' %
                      (classif_cost, ':'.join(['%d' % c for c in corr]),
                       len(inst_nos), ':'.join(['%.3f' % d for d in dist])))

            pass_valid += valid
            pass_dist += dist
            pass_corr += corr
            pass_classif_cost += classif_cost

            # ranking evaluation
            rank_results = outputs[1]
            corr_ranks = np.sum((rank_results > 0.0).astype(np.float) * ranking_mask, axis=0).astype(np.int)
            pass_corr_ranks += corr_ranks
            pass_valid_ranks += np.sum(ranking_mask, axis=0)
            pass_ranking_cost += ranking_cost

            pass_cost += total_cost

        # print and return statistics
        self._print_pass_stats(pass_no, datetime.timedelta(seconds=(time.time() - pass_start_time)),
                               pass_classif_cost, pass_corr, pass_dist, pass_valid,
                               pass_ranking_cost, pass_corr_ranks, pass_valid_ranks)
        return pass_cost

    def _print_pass_stats(self, pass_no, time, cost, corr, dist, n_inst,
                          rank_cost, rank_corr, rank_n_inst):

        acc = ':'.join(['%.3f' % a for a in (corr / n_inst)])
        rank_acc = ':'.join(['%.3f' % a for a in (rank_corr / rank_n_inst)])
        avg_dist = ':'.join(['%.3f' % d for d in (dist / n_inst)])
        log_info('PASS %03d: duration %s, cost %f, rcost %f, acc %s, racc %s, avg. dist %s' %
                 (pass_no, str(time), cost, rank_cost, acc, rank_acc, avg_dist))
        self.tb_logger.add_to_log(pass_no, {'train_pass_duration': time.total_seconds(),
                                            'train_classif_cost': cost,
                                            'train_accuracy': corr / n_inst,
                                            'train_dist_avg': dist / n_inst,
                                            'train_rank_acc': rank_corr / rank_n_inst,
                                            'train_rank_cost': rank_cost})

    def _print_valid_stats(self, pass_no, results):
        """Print validation results for the given training pass number."""
        aspects = sorted([aspect for aspect in results.iterkeys()
                          if isinstance(results[aspect], dict)])
        to_print = []
        for quant in ['dist_total', 'dist_avg', 'accuracy', 'pearson', 'spearman']:
            to_print.append(':'.join(['%.3f' % results[aspect][quant] for aspect in aspects]))
        to_print.append(results['cost_comb'])

        log_info(('Validation distance: %s (avg: %s), accuracy %s, ' +
                  'pearson %s, spearman %s, combined cost %.3f') % tuple(to_print))

        for aspect in aspects:
            for key, val in results[aspect].iteritems():
                results[aspect + '_' + key] = val
            del results[aspect]
        self.tb_logger.add_to_log(pass_no, {'valid_' + key: value
                                            for key, value in results.iteritems()})

    def _seq2seq_training_pass(self, pass_no):
        pass_start_time = time.time()

        log_debug('\n***\nTR %05d:' % pass_no)
        pass_cost = 0.0

        for inst_nos in self._seq2seq_train_batches():

            log_debug('INST-NOS: ' + str(inst_nos))
            log_debug("\n".join(' '.join([tok for tok, _ in self.train_hyps[i]]) + "\n" +
                                ' '.join([tok for tok, _ in self.train_refs[i]]) + "\n" +
                                unicode(self.train_das[i])
                                for i in inst_nos))

            fd = {}
            self._add_inputs_to_feed_dict(fd,
                                          inputs_hyp=self.X_hyp[inst_nos],
                                          inputs_da=self.X_da[inst_nos],
                                          inputs_s2s_trg=True, train_mode=True)

            required = [self.s2s_cost, self.s2s_train_func]
            cost, _ = self.session.run(required, feed_dict=fd)
            log_debug('COST: %f' % cost)

            pass_cost += cost

        # print and return statistics
        self._print_seq2seq_pass_stats(pass_no,
                                       datetime.timedelta(seconds=(time.time() - pass_start_time)),
                                       pass_cost)
        return pass_cost

    def _seq2seq_evaluate(self, inputs, targets):
        """Evaluate seq2seq model for next-word prediction on the given data (typically validation
        data). Will only use top 1/4-rated instances.
        @param inputs: 3-tuple of validation DAs, references (unused), hypotheses
        @param targets: human scores of the DA-hypotheses pairs
        @return: the total decoder cost of decoding the given hypotheses given the input DAs
        """
        das, _, hyps = inputs
        # convert inputs and outputs
        hyps = np.array([self.embs.get_embeddings(sent) for sent in hyps])
        das = np.array([self.da_embs.get_embeddings(da) for da in das])
        y = np.array(targets)

        # run and calculate cost
        cost = 0.0
        for inst_nos in self._seq2seq_batches(range(len(inputs)), y):

            fd = {}
            self._add_inputs_to_feed_dict(fd,
                                          inputs_hyp=hyps[inst_nos],
                                          inputs_da=das[inst_nos],
                                          inputs_s2s_trg=True)
            cost += self.session.run([self.s2s_cost], feed_dict=fd)[0]
        return cost

    def _print_seq2seq_pass_stats(self, pass_no, time, pass_cost):
        log_info('PASS %03d: duration %s, cost %f' % (pass_no, str(time), pass_cost))

    def _print_seq2seq_validation_stats(self, pass_no, pass_cost):
        log_info('PASS %03d: seq2seq validation cost: %f' % (pass_no, pass_cost))

    def _seq2seq_train_batches(self):
        # TODO this probably won't work for binary predictors (but they're unused anyway)
        return self._seq2seq_batches(self.train_order, self.y)

    def _seq2seq_batches(self, order, targets):
        """Create batches from the input (using only "good" instances); use as iterator."""
        # find out which are the best instances
        if self.scale_reversed:
            good_lo = self.outputs_range_lo
            good_hi = good_lo + ((self.outputs_range_hi - self.outputs_range_lo) / 4.0)
        else:
            good_hi = self.outputs_range_hi
            good_lo = good_hi - ((self.outputs_range_hi - self.outputs_range_lo) / 4.0)
        # filter instances, use best 1/4 of the range
        order = [i for i in order
                 if targets[i] >= good_lo and targets[i] <= good_hi]
        # yield only filtered instances
        for i in xrange(0, len(order), self.batch_size):
            yield order[i: i + self.batch_size]

    def evaluate(self, inputs, raw_targets, output_file=None):
        """
        Evaluate the predictor on the given inputs & targets; possibly also write to a file.
        """
        # XXX TODO fix for hyp2s -- should provide ranking for these
        if isinstance(inputs, tuple):
            das, input_refs, input_hyps = inputs[:3]  # ignore is_real indicators
        else:
            das, input_refs, input_hyps = self._divide_inputs(inputs)
        evaler = Evaluator(self.target_cols)
        for da, input_ref, input_hyp, raw_target in zip(das, input_refs, input_hyps, raw_targets):
            raw_rating = self.rate([input_hyp] if self.hyp_enc else None,
                                   [input_ref] if self.ref_enc else None,
                                   [da] if self.da_enc else None)[0]  # remove the "batch" dimension
            rating = self._round_rating(raw_rating)
            target = self._round_rating(raw_target)
            evaler.append((da, input_ref, input_hyp),
                          raw_target, target, raw_rating, rating)
        if output_file:
            evaler.write_tsv(output_file)
        return evaler.get_stats()

    def _daclassif_training_pass(self, pass_no):
        pass_start_time = time.time()

        log_debug('\n***\nTR %05d:' % pass_no)
        pass_cost = 0.0

        for inst_nos in self._seq2seq_train_batches():

            log_debug('INST-NOS: ' + str(inst_nos))
            log_debug("\n".join(' '.join([tok for tok, _ in self.train_hyps[i]]) + "\n" +
                                unicode(self.train_das[i])
                                for i in inst_nos))

            fd = {}
            self._add_inputs_to_feed_dict(fd,
                                          inputs_hyp=self.X_hyp[inst_nos],
                                          inputs_daclassif_trg=self.daclassif_y[inst_nos],
                                          train_mode=True)

            required = [self.daclassif_cost, self.daclassif_train_func]
            cost, _ = self.session.run(required, feed_dict=fd)
            log_debug('COST: %f' % cost)

            pass_cost += cost

        # print and return statistics
        self._print_daclassif_pass_stats(
            pass_no, datetime.timedelta(seconds=(time.time() - pass_start_time)), pass_cost)
        return pass_cost

    def _daclassif_evaluate(self, inputs, ratings):
        """Evaluate DA classification on the given data (typically validation data).
        Will only use top 1/4-rated instances.
        @param inputs: 3-tuple of validation DAs, references (unused), hypotheses
        @param ratings: human scores of the DA-hypotheses pairs (use for filtering)
        @return: the total decoder cost of decoding the given hypotheses given the input DAs
        """
        das, _, hyps = inputs
        # convert inputs and outputs
        hyps = np.array([self.embs.get_embeddings(sent) for sent in hyps])
        das = [self.daclassif_feats.get_features(None, {'da': da}) for da in das]
        das = self.daclassif_vect.transform(das)
        ratings = np.array(ratings)

        # run and calculate cost
        cost = 0.0
        for inst_nos in self._seq2seq_batches(range(len(inputs)), ratings):

            fd = {}
            self._add_inputs_to_feed_dict(fd,
                                          inputs_hyp=hyps[inst_nos],
                                          inputs_daclassif_trg=das[inst_nos])
            cost += self.session.run([self.daclassif_cost], feed_dict=fd)[0]
        return cost

    def _print_daclassif_pass_stats(self, pass_no, time, pass_cost):
        log_info('PASS %03d: duration %s, cost %f' % (pass_no, str(time), pass_cost))

    def _print_daclassif_validation_stats(self, pass_no, pass_cost):
        log_info('PASS %03d: DA classification validation cost: %f' % (pass_no, pass_cost))
