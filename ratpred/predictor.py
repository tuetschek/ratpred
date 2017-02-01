#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Classifying trees to determine which DAIs are represented.
"""

from __future__ import unicode_literals
import cPickle as pickle
import time
import datetime
import sys
import re
import math
import os.path

import numpy as np
import scipy.stats
import tensorflow as tf

from pytreex.core.util import file_stream

from tgen.logf import log_info, log_debug
from tgen.rnd import rnd
from tgen.embeddings import TokenEmbeddingSeq2SeqExtract, DAEmbeddingSeq2SeqExtract
from tgen.tf_ml import TFModel

from ratpred.futil import read_data, write_outputs
from ratpred.embeddings import Word2VecEmbeddingExtract, CharEmbeddingExtract
from ratpred.tb_logging import DummyTensorBoardLogger, TensorBoardLogger


def sigmoid(nums):
    return 1 / (1 + np.exp(-nums))


class RatingPredictor(TFModel):

    def __init__(self, cfg):

        super(RatingPredictor, self).__init__(scope_name='predict_rating-' +
                                              cfg.get('scope_suffix', ''))
        self.cfg = cfg
        self.emb_size = cfg.get('emb_size', 50)
        self.cell_type = cfg.get('cell_type', 'lstm')
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

        self.target_col = cfg.get('target_col', 'quality')
        self.delex_slots = cfg.get('delex_slots', set())
        if self.delex_slots:
            self.delex_slots = set(self.delex_slots.split(','))
        self.delex_slot_names = cfg.get('delex_slot_names', False)
        self.reuse_embeddings = cfg.get('reuse_embeddings', False)
        self.tanh_layers = cfg.get('tanh_layers', 0)
        self.predict_ints = cfg.get('predict_ints', False)
        self.predict_halves = cfg.get('predict_halves', False)
        self.predict_coarse = cfg.get('predict_coarse', None)
        self.num_outputs = 1  # will be changed in training if predict_ints is True

        self.tb_logger = DummyTensorBoardLogger()

    def set_tensorboard_logging(self, log_dir, run_id):
        self.tb_logger = TensorBoardLogger(log_dir, run_id)

    def save_to_file(self, model_fname, skip_settings=False):
        """Save the predictor to a file (actually two files, one for configuration and one
        for the TensorFlow graph, which must be stored separately).

        @param model_fname: file name (for the configuration file); TF graph will be stored with a \
            different extension
        @param skip_settings: if True, saving settings is skipped, only the model parameters are \
            written to disk
        """
        if not skip_settings:
            log_info("Saving model settings to %s..." % model_fname)
            with file_stream(model_fname, 'wb', encoding=None) as fh:
                pickle.dump(self.get_all_settings(), fh, protocol=pickle.HIGHEST_PROTOCOL)
        tf_session_fname = re.sub(r'(.pickle)?(.gz)?$', '.tfsess', model_fname)
        log_info("Saving model parameters to %s..." % tf_session_fname)
        self.saver.save(self.session, tf_session_fname)

    def get_all_settings(self):
        """Get all settings except the trained model parameters (to be stored in a pickle)."""
        data = {'cfg': self.cfg,
                'embs': self.embs, }
        if self.embs:
            data['dict_size'] = self.dict_size
            data['input_shape'] = self.input_shape
            data['outputs_range_lo'] = self.outputs_range_lo
            data['outputs_range_hi'] = self.outputs_range_hi
            if self.predict_ints:
                data['num_outputs'] = self.num_outputs
        if self.da_enc:
            data['da_embs'] = self.da_embs
            data['da_dict_size'] = self.da_dict_size
            data['da_input_shape'] = self.da_input_shape
        return data

    def _save_checkpoint(self):
        """Store an in-memory checkpoint containing all variables and settings of the model.
        Will always overwrite the last checkpoint."""
        log_info('Storing in-memory checkpoint...')
        self.checkpoint = (self.get_all_settings(), self.get_model_params())

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

        # re-build TF graph and restore the TF session
        tf_session_fname = os.path.abspath(re.sub(r'(.pickle)?(.gz)?$', '.tfsess', model_fname))
        ret._init_neural_network()
        ret.saver.restore(ret.session, tf_session_fname)
        return ret

    def load_data(self, data_file):
        """Load a data file, return inputs and targets."""
        return read_data(data_file, self.target_col,
                         'text' if self.da_enc == 'token' else 'cambridge',
                         self.delex_slots, self.delex_slot_names)

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

        # start training
        top_cost = float('nan')

        for iter_no in xrange(1, self.passes + 1):
            self.train_order = range(len(self.train_hyps))
            if self.randomize:
                rnd.shuffle(self.train_order)
            pass_cost = self._training_pass(iter_no)

            if (self.valid_inputs and self.validation_freq and
                    iter_no > self.min_passes and iter_no % self.validation_freq == 0):

                results = self.evaluate(self.valid_inputs, self.valid_y)
                results['cost_total'] = pass_cost
                results['cost_avg'] = pass_cost / len(self.train_hyps)
                self._compute_comb_cost(results)
                self._print_valid_stats(iter_no, results)

                # if we have the best model so far, save it as a checkpoint (overwrite previous)
                if math.isnan(top_cost) or results['cost_comb'] < top_cost:
                    top_cost = results['cost_comb']
                    self._save_checkpoint()
                    self.checkpoint_pass = iter_no

                    # once in a while, save current best checkpoint to disk
                    if (model_fname and
                            (iter_no > self.disk_store_min_pass) and
                            (iter_no - self.disk_stored_pass >= self.disk_store_freq)):
                        self.save_to_file(model_fname, self.disk_stored_pass > 0)
                        self.disk_stored_pass = iter_no

            if iter_no == self.pretrain_passes:  # continue training only on real data
                self._remove_fake_training_data()

        # restore last checkpoint (best performance on devel data)
        self._restore_checkpoint()

    def _remove_fake_training_data(self):
        """Remove 'fake' training data from the list of training instances, based on the
        self.train_is_reals variable."""
        def filter_real(data):
            return [inst for inst, ir in zip(data, self.train_is_reals) if ir]

        self.train_das = filter_real(self.train_das)
        self.train_hyps = filter_real(self.train_hyps)
        self.train_refs = filter_real(self.train_refs)
        self.y = np.array(filter_real(self.y))
        self.train_is_reals = filter_real(self.train_is_reals)  # basically set all to 1

    def _compute_comb_cost(self, results):
        """Compute combined cost, given my validation quantity weights."""
        comb_cost = sum([weight * results[quantity]
                         for quantity, weight in self.validation_weights.iteritems()])
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
        # TODO possibly need to transpose the output here as well
        # TODO the rest does not support batches even if the previous does !!!
        val = self.session.run(self.output, feed_dict=fd)
        return float(self._adjust_output(val))

    def _adjust_output(self, val, no_sigmoid=False):
        if self.predict_ints:
            # do the actual sigmoid + squeeze it into our range (using defined coarseness)
            coeff = self._rounding_step('train')
            if not no_sigmoid:
                val = sigmoid(val)
            return np.clip(coeff * np.sum(val, axis=1, keepdims=True) + self.outputs_range_lo,
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
        if get_is_real:
            irs = [inp[3] for inp in inputs[:size]]
            return (das, refs, hyps, irs)
        return (das, refs, hyps)

    def _cut_valid_data(self):
        self.valid_inputs = (self.train_das[-self.validation_size:],
                             self.train_refs[-self.validation_size:],
                             self.train_hyps[-self.validation_size:])
        self.valid_y = self.y[-self.validation_size:]
        self.y = self.y[:-self.validation_size]
        self.train_das = self.train_das[:-self.validation_size]
        self.train_refs = self.train_refs[:-self.validation_size]
        self.train_hyps = self.train_hyps[:-self.validation_size]
        self.train_is_reals = self.train_is_reals[:-self.validation_size]

    def _init_training(self, inputs, targets,
                       valid_inputs=None, valid_targets=None, data_portion=1.0):
        """Initialize training.

        @param data_portion: portion of the training data to be used (0.0-1.0)
        """
        # store training data, make it smaller if necessary
        train_size = int(round(data_portion * len(inputs)))
        self.train_das, self.train_refs, self.train_hyps, self.train_is_reals = (
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

        # initialize input embeddings
        if self.hyp_enc:
            self.dict_size = self.embs.init_dict(self.train_hyps)
        else:
            self.dict_size = len(self.embs.dict) + 1
        if self.ref_enc:
            self.dict_size = self.embs.init_dict(self.train_refs, dict_ord=self.dict_size)
        if self.da_enc:
            self.da_dict_size = self.da_embs.init_dict(self.train_das)

        # convert training data to indexes
        if self.hyp_enc:
            self.X_hyp = np.array([self.embs.get_embeddings(sent) for sent in self.train_hyps])
        if self.ref_enc:
            self.X_ref = np.array([self.embs.get_embeddings(sent) for sent in self.train_refs])
        if self.da_enc:
            self.X_da = np.array([self.da_embs.get_embeddings(da) for da in self.train_das])

        # initialize I/O shapes and boundaries
        self.input_shape = self.embs.get_embeddings_shape()
        if self.da_enc:
            self.da_input_shape = self.da_embs.get_embeddings_shape()

        self.outputs_range_lo = int(round(min(self.y)))
        self.outputs_range_hi = int(round(max(self.y)))
        if self.predict_ints:
            self.y = self._ratings_to_binary(self.y)
            if self.predict_coarse == 'train':
                self.num_outputs = 3
            else:
                # we actually want 1 output less than the range (all 0's = lo, all 1'= hi)
                self.num_outputs = self.outputs_range_hi - self.outputs_range_lo
                if self.predict_halves:
                    # all 1/2 steps between hi and lo, i.e., 2*(range-1)
                    self.num_outputs *= 2
        else:
            # make target output 1-D and round it to desired coarseness
            self.y = np.array([[self._round_rating(y_, mode='train')] for y_ in self.y])
            self.num_outputs = 1  # just one real-valued output

        # initialize NN classifier
        self._init_neural_network()
        # initialize the NN variables
        self.session.run(tf.global_variables_initializer())

    def _rounding_step(self, mode='test'):
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
        the (half-)int, 0's higher)."""
        step = self._rounding_step('train')
        ints = [[0 if val < i + (step / 2.0) else 1
                 for i in np.arange(self.outputs_range_lo, self.outputs_range_hi, step)]
                for val in ints]
        return np.array(ints)

    def _init_neural_network(self):
        """Create the neural network for classification"""

        # set TensorFlow random seed
        tf.set_random_seed(rnd.randint(-sys.maxint, sys.maxint))

        self.target = tf.placeholder(tf.float32, [None, self.num_outputs], name='target')

        with tf.variable_scope(self.scope_name):

            self.train_mode = tf.placeholder(tf.bool, [], name='train_mode')

            if self.hyp_enc:
                self.initial_state_hyp = tf.placeholder(tf.float32, [None, self.emb_size],
                                                        name='enc_inp_hyp_init')
                self.inputs_hyp = [tf.placeholder(tf.int32, [None], name=('enc_inp_hyp-%d' % i))
                                   for i in xrange(self.input_shape[0])]
            if self.ref_enc:
                self.initial_state_ref = tf.placeholder(tf.float32, [None, self.emb_size],
                                                        name='enc_inp_ref_init')
                self.inputs_ref = [tf.placeholder(tf.int32, [None], name=('enc_inp_ref-%d' % i))
                                   for i in xrange(self.input_shape[0])]
            if self.da_enc:
                self.initial_state_da = tf.placeholder(tf.float32, [None, self.emb_size],
                                                       name='enc_inp_da_init')
                self.inputs_da = [tf.placeholder(tf.int32, [None], name=('enc_inp_da-%d' % i))
                                  for i in xrange(self.da_input_shape[0])]

            if self.cell_type.startswith('gru'):
                self.cell = tf.nn.rnn_cell.GRUCell(self.emb_size)
            else:
                self.cell = tf.nn.rnn_cell.BasicLSTMCell(self.emb_size)
            if self.cell_type.endswith('/2'):
                self.cell = tf.nn.rnn_cell.MultiRNNCell([self.cell] * 2)

            self.output = self._classif_net(self.inputs_hyp if self.hyp_enc else None,
                                            self.inputs_ref if self.ref_enc else None,
                                            self.inputs_da if self.da_enc else None)

        if self.predict_ints:
            # sigmoid cost -- predict a bunch of 1's and 0's (not just one 1)
            self.cost = tf.reduce_mean(tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(self.output, self.target, name='CE'),
                axis=1))
        else:
            # mean square error cost -- predict 1 number
            # NB: needs to compute mean over axis=0 only, otherwise it won't work (?)
            self.cost = tf.reduce_mean(tf.square(self.target - self.output), axis=0)

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

        # this helps us load/save the model
        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2)

    def _dropout(self, variable):
        if self.dropout_keep_prob == 1.0:
            return variable
        train_mode_mask = tf.fill(tf.shape(variable)[:1], self.train_mode)
        return tf.select(train_mode_mask,
                         tf.nn.dropout(variable, self.dropout_keep_prob),
                         variable)

    def _classif_net(self, enc_inputs_hyp=None, enc_inputs_ref=None, enc_inputs_da=None):
        """Build the rating prediction RNN structure.
        @return: TensorFlow Output with the prediction
        """
        # build embeddings
        with tf.variable_scope('embs') as scope:
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

        # apply RNN over embeddings
        if enc_inputs_hyp is not None:
            with tf.variable_scope('enc_hyp') as scope:
                enc_in_hyp_emb = [apply_emb(enc_inp) for enc_inp in enc_inputs_hyp]
                enc_outs_hyp, enc_state_hyp = tf.nn.rnn(self.cell, enc_in_hyp_emb, dtype=tf.float32)

        if enc_inputs_ref is not None:
            if self.reuse_embeddings and enc_inputs_hyp is not None:
                scope = tf.variable_scope('enc_hyp')
                scope.reuse_variables()
            else:
                scope = tf.variable_scope('enc_ref')

            with scope:
                enc_in_ref_emb = [apply_emb(enc_inp) for enc_inp in enc_inputs_ref]
                enc_outs_ref, enc_state_ref = tf.nn.rnn(self.cell, enc_in_ref_emb, dtype=tf.float32)

        if enc_inputs_da is not None:
            with tf.variable_scope('enc_da'):
                sqrt3 = math.sqrt(3)
                self.da_emb_storage = tf.get_variable(
                    'emb_storage',
                    (self.da_dict_size, self.emb_size),
                    initializer=tf.random_uniform_initializer(-sqrt3, sqrt3))
                enc_in_da_emb = [self._dropout(tf.nn.embedding_lookup(self.da_emb_storage, enc_inp))
                                 for enc_inp in enc_inputs_da]
                enc_outs_da, enc_state_da = tf.nn.rnn(self.cell, enc_in_da_emb, dtype=tf.float32)
                self.tb_logger.create_tensor_summaries(self.da_emb_storage)

        # concatenate last LSTM states & outputs (works for multilayer LSTMs&GRUs)
        last_outs_and_states = tf.concat(1, (self._flatten_enc_state(enc_state_hyp)
                                             if enc_inputs_hyp is not None else []) +
                                         (self._flatten_enc_state(enc_state_ref)
                                          if enc_inputs_ref is not None else []) +
                                         (self._flatten_enc_state(enc_state_da)
                                          if enc_inputs_da is not None else []))
        state_size = int(last_outs_and_states.get_shape()[1])
        hidden = last_outs_and_states
        self.tb_logger.create_tensor_summaries(hidden)

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

        with tf.variable_scope('classif'):
            w = tf.get_variable('final-transf-w', (state_size, self.num_outputs),
                                initializer=tf.random_normal_initializer(stddev=0.1))
            b = tf.get_variable('final-transf-b', (self.num_outputs,),
                                initializer=tf.constant_initializer())
            self.tb_logger.create_tensor_summaries(w)
            self.tb_logger.create_tensor_summaries(b)

        return tf.matmul(hidden, w) + b

    def _batches(self):
        """Create batches from the input; use as iterator."""
        for i in xrange(0, len(self.train_order), self.batch_size):
            yield self.train_order[i: i + self.batch_size]

    def _flatten_enc_state(self, enc_state):
        """Flatten up to two dimensions of tuples, return 1-D array."""
        if isinstance(enc_state, tuple):
            if isinstance(enc_state[0], tuple):
                return [x for y in enc_state for x in y]
            return [x for x in enc_state]
        return [enc_state]

    def _add_inputs_to_feed_dict(self, fd,
                                 inputs_hyp=None, inputs_ref=None, inputs_da=None,
                                 train_mode=False):

        fd[self.train_mode] = train_mode

        if inputs_hyp is not None:
            fd[self.initial_state_hyp] = np.zeros([inputs_hyp.shape[0], self.emb_size])
            # TODO check for none when squeezing ?
            sliced_hyp = np.squeeze(np.array(np.split(inputs_hyp, len(inputs_hyp[0]), axis=1)), axis=2)
            for input_, slice_ in zip(self.inputs_hyp, sliced_hyp):
                fd[input_] = slice_

        if inputs_ref is not None:
            fd[self.initial_state_ref] = np.zeros([inputs_ref.shape[0], self.emb_size])
            sliced_ref = np.squeeze(np.array(np.split(inputs_ref, len(inputs_ref[0]), axis=1)), axis=2)
            for input_, slice_ in zip(self.inputs_ref, sliced_ref):
                fd[input_] = slice_

        if inputs_da is not None:
            fd[self.initial_state_da] = np.zeros([inputs_da.shape[0], self.emb_size])
            sliced_da = np.squeeze(np.array(np.split(inputs_da, len(inputs_da[0]), axis=1)), axis=2)
            for input_, slice_ in zip(self.inputs_da, sliced_da):
                fd[input_] = slice_

    def _training_pass(self, pass_no):
        """Perform one training pass through the whole training data, print statistics."""

        pass_start_time = time.time()

        log_debug('\n***\nTR %05d:' % pass_no)
        log_debug("Train order: " + str(self.train_order))

        pass_insts = 0
        pass_cost = 0
        pass_corr = 0
        pass_dist = 0

        for inst_nos in self._batches():

            pass_insts += len(inst_nos)
            log_debug('INST-NOS: ' + str(inst_nos))
            log_debug("\n".join(' '.join([tok for tok, _ in self.train_hyps[i]]) + "\n" +
                                ' '.join([tok for tok, _ in self.train_refs[i]]) + "\n" +
                                unicode(self.train_das[i]) + "\n" +
                                unicode(self.y[i])
                                for i in inst_nos))

            fd = {self.target: self.y[inst_nos]}
            self._add_inputs_to_feed_dict(fd, self.X_hyp[inst_nos] if self.hyp_enc else None,
                                          self.X_ref[inst_nos] if self.ref_enc else None,
                                          self.X_da[inst_nos] if self.da_enc else None,
                                          True)

            required = [self.output, self.cost, self.train_func]
            if pass_insts == len(self.train_hyps):  # last batch
                required.append(self.tensor_summaries)
            outputs = self.session.run(required, feed_dict=fd)
            results = outputs[0]
            cost = outputs[1]
            if pass_insts == len(self.train_hyps):
                self.tb_logger.log(pass_no, outputs[3])

            pred = self._adjust_output(results)
            true = self._adjust_output(self.y[inst_nos], no_sigmoid=True)
            dist = np.sum(np.abs(pred - true))
            corr = np.sum(self._round_rating(pred) == self._round_rating(true))

            log_debug('R: ' + str(results))
            log_debug('COST: %f, corr %d/%d, dist %.3f' % (cost, corr, len(inst_nos), dist))

            pass_dist += dist
            pass_corr += corr
            pass_cost += cost

        # print and return statistics
        self._print_pass_stats(pass_no, datetime.timedelta(seconds=(time.time() - pass_start_time)),
                               pass_cost,
                               float(pass_corr) / len(self.train_hyps),
                               pass_dist / len(self.train_hyps))

        return pass_cost

    def _print_pass_stats(self, pass_no, time, cost, acc, avg_dist):
        log_info('PASS %03d: duration %s, cost %f, acc %.3f, avg. dist %.3f' %
                 (pass_no, str(time), cost, acc, avg_dist))
        self.tb_logger.add_to_log(pass_no, {'train_pass_duration': time.total_seconds(),
                                            'train_cost': cost,
                                            'train_accuracy': acc,
                                            'train_dist_avg': avg_dist})

    def _print_valid_stats(self, pass_no, results):
        """Print validation results for the given training pass number."""
        log_info(('Validation distance: %.3f (avg: %.3f), accuracy %.3f, ' +
                  'pearson %.3f, combined cost %.3f') %
                 tuple(results[key]
                       for key in ['dist_total', 'dist_avg', 'accuracy', 'pearson', 'cost_comb']))
        self.tb_logger.add_to_log(pass_no, {'valid_' + key: value
                                            for key, value in results.iteritems()})

    def evaluate(self, inputs, raw_targets, output_file=None):
        """
        Evaluate the predictor on the given inputs & targets; possibly also write to a file.
        """
        if isinstance(inputs, tuple):
            das, input_refs, input_hyps = inputs[:3]  # ignore is_real indicators
        else:
            das, input_refs, input_hyps = self._divide_inputs(inputs)
        dists = []
        correct = 0
        raw_ratings = []
        ratings = []
        targets = []
        for da, input_ref, input_hyp, target in zip(das, input_refs, input_hyps, raw_targets):
            rating = self.rate([input_hyp] if self.hyp_enc else None,
                               [input_ref] if self.ref_enc else None,
                               [da] if self.da_enc else None)
            raw_ratings.append(rating)
            dists.append(abs(rating - target))  # calculate distance on raw ratings & targets
            rating = self._round_rating(rating)  # calculate accuracy & correlation on rounded
            target = self._round_rating(target)
            ratings.append(rating)
            targets.append(target)
            if rating == target:
                correct += 1
        pearson, pearson_pv = scipy.stats.pearsonr(targets, ratings)
        spearman, spearman_pv = scipy.stats.spearmanr(targets, ratings)
        if output_file:
            write_outputs(output_file, inputs, raw_targets, targets, raw_ratings, ratings)
        return {'dist_total': np.sum(dists),
                'dist_avg': np.mean(dists),
                'dist_stddev': np.std(dists),
                'accuracy': float(correct) / len(input_hyps),
                'pearson': pearson,
                'pearson_pv': pearson_pv,
                'spearman': spearman,
                'spearman_pv': spearman_pv}
