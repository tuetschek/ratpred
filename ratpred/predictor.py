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
from ratpred.embeddings import Word2VecEmbeddingExtract


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
        if 'word2vec_model' in cfg:
            self.embs = Word2VecEmbeddingExtract(cfg)
        else:
            self.embs = TokenEmbeddingSeq2SeqExtract(cfg)

        self.da_enc = cfg.get('da_enc', False)
        if self.da_enc:
            self.da_embs = DAEmbeddingSeq2SeqExtract(cfg)

        self.passes = cfg.get('passes', 200)
        self.min_passes = cfg.get('min_passes', 0)
        self.alpha = cfg.get('alpha', 0.1)
        self.randomize = cfg.get('randomize', True)
        self.batch_size = cfg.get('batch_size', 1)

        self.validation_size = cfg.get('validation_size', 0)
        self.validation_freq = cfg.get('validation_freq', 10)
        self.max_cores = cfg.get('max_cores')
        self.checkpoint = None

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

    def save_to_file(self, model_fname):
        """Save the predictor to a file (actually two files, one for configuration and one
        for the TensorFlow graph, which must be stored separately).

        @param model_fname: file name (for the configuration file); TF graph will be stored with a \
            different extension
        """
        log_info("Saving classifier to %s..." % model_fname)
        with file_stream(model_fname, 'wb', encoding=None) as fh:
            pickle.dump(self.get_all_settings(), fh, protocol=pickle.HIGHEST_PROTOCOL)
        tf_session_fname = re.sub(r'(.pickle)?(.gz)?$', '.tfsess', model_fname)
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
        if self.da_embs:
            data['da_embs'] = self.da_embs
            data['da_dict_size'] = self.da_dict_size
        return data

    def _save_checkpoint(self):
        """Store an in-memory checkpoint containing all variables and settings of the model.
        Will always overwrite the last checkpoint."""
        log_info('Storing in-memory checkpoint...')
        self.checkpoint = (self.get_all_settings(), self.get_model_params())

    def restore_checkpoint(self):
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

    def train(self, train_data_file, valid_data_file=None, data_portion=1.0):
        """Run training on the given training data.
        """
        inputs, targets = read_data(train_data_file, self.target_col,
                                    self.delex_slots, self.delex_slot_names)
        valid_inputs, valid_targets = None, None
        if valid_data_file:
            valid_inputs, valid_targets = read_data(valid_data_file, self.target_col,
                                                    self.delex_slots, self.delex_slot_names)
        log_info('Training rating predictor...')

        # initialize training
        self._init_training(inputs, targets, valid_inputs, valid_targets, data_portion)

        # start training
        top_cost = float('nan')

        for iter_no in xrange(1, self.passes + 1):
            self.train_order = range(len(self.train_refs))
            if self.randomize:
                rnd.shuffle(self.train_order)
            pass_cost = self._training_pass(iter_no)

            if (self.valid_inputs and self.validation_freq and
                    iter_no > self.min_passes and iter_no % self.validation_freq == 0):

                results = self.evaluate(self.valid_inputs, self.valid_y)
                log_info('Validation distance: %.3f (avg: %.3f), accuracy %.3f' %
                         (results['dist_total'], results['dist_avg'], results['accuracy']))

                comb_cost = len(self.valid_inputs) * results['accuracy'] + 100 * results['dist_total'] + pass_cost
                log_info('Combined validation cost: %.3f' % comb_cost)

                # if we have the best model so far, save it as a checkpoint (overwrite previous)
                if math.isnan(top_cost) or comb_cost < top_cost:
                    top_cost = comb_cost
                    self._save_checkpoint()

        # restore last checkpoint (best performance on devel data)
        self.restore_checkpoint()

    def rate(self, refs, hyps, das=None):
        """
        Rate a pair of reference sentence + system output hypothesis.

        @param refs: a reference sentence (as a 1-element array, batches not yet supported)
        @param hyps: a system output hypothesis (as a 1-element array, batches not yet supported)
        @return: the rating, as a floating point number (not rounded to prediction boundaries)
        """
        inputs_ref = np.array([self.embs.get_embeddings(sent) for sent in refs])
        inputs_hyp = np.array([self.embs.get_embeddings(sent) for sent in hyps])
        if das:
            inputs_da = np.array([self.da_embs.get_embeddings(da) for da in das])
        fd = {}
        self._add_inputs_to_feed_dict(fd, inputs_ref, inputs_hyp, inputs_da)
        # TODO possibly need to transpose the output here as well
        # TODO the rest does not support batches even if the previous does !!!
        val = self.session.run(self.output, feed_dict=fd)
        if self.predict_ints:
            # do the actual sigmoid + squeeze it into our range (using defined coarseness)
            coeff = self._rounding_step('train')
            return min(coeff * np.sum(sigmoid(val)) + self.outputs_range_lo, self.outputs_range_hi)
        else:
            # just squeeze the output float value into our range
            return max(float(self.outputs_range_lo), min(float(self.outputs_range_hi), float(val)))

    def _divide_inputs(self, inputs, trunc_size=None):
        size = trunc_size if trunc_size is not None else len(inputs)
        return ([da for da, _, _ in inputs[:size]],
                [ref for _, ref, _ in inputs[:size]],
                [hyp for _, _, hyp in inputs[:size]])

    def _cut_valid_data(self):
        self.valid_inputs = (self.train_das[-self.validation_size:],
                             self.train_refs[-self.validation_size:],
                             self.train_hyps[-self.validation_size:])
        self.valid_y = self.y[-self.validation_size:]
        self.y = self.y[:-self.validation_size]
        self.train_das = self.train_das[:-self.validation_size]
        self.train_refs = self.train_refs[:-self.validation_size]
        self.train_hyps = self.train_hyps[:-self.validation_size]

    def _init_training(self, inputs, targets,
                       valid_inputs=None, valid_targets=None, data_portion=1.0):
        """Initialize training.

        @param data_portion: portion of the training data to be used (0.0-1.0)
        """
        # store training data, make it smaller if necessary
        train_size = int(round(data_portion * len(inputs)))
        self.train_das, self.train_refs, self.train_hyps = self._divide_inputs(inputs, train_size)
        self.y = targets[:train_size]

        self.valid_inputs, self.valid_y = None, None
        if valid_inputs is not None and valid_targets is not None:
            self.valid_inputs = self._divide_inputs(valid_inputs)
            self.valid_y = valid_targets
        elif self.validation_size > 0:
            self._cut_valid_data()

        self.train_order = range(len(self.train_refs))
        log_info('Using %d training instances.' % len(self.train_refs))

        # initialize input embeddings
        self.dict_size = self.embs.init_dict(self.train_refs)
        self.dict_size = self.embs.init_dict(self.train_hyps, dict_ord=self.dict_size)
        if self.da_enc:
            self.da_dict_size = self.da_embs.init_dict(self.train_das)

        # convert training data to indexes
        self.X_ref = np.array([self.embs.get_embeddings(sent) for sent in self.train_refs])
        self.X_hyp = np.array([self.embs.get_embeddings(sent) for sent in self.train_hyps])
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
        return self.outputs_range_lo + round((rating - self.outputs_range_lo) / step) * step

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

            self.initial_state_ref = tf.placeholder(tf.float32, [None, self.emb_size])
            self.initial_state_hyp = tf.placeholder(tf.float32, [None, self.emb_size])
            self.inputs_ref = [tf.placeholder(tf.int32, [None], name=('enc_inp_ref-%d' % i))
                               for i in xrange(self.input_shape[0])]
            self.inputs_hyp = [tf.placeholder(tf.int32, [None], name=('enc_inp_hyp-%d' % i))
                               for i in xrange(self.input_shape[0])]
            if self.da_enc:
                self.initial_state_da = tf.placeholder(tf.float32, [None, self.emb_size])
                self.inputs_da = [tf.placeholder(tf.int32, [None], name=('enc_inp_da-%d' % i))
                                  for i in xrange(self.da_input_shape[0])]

            if self.cell_type.startswith('gru'):
                self.cell = tf.nn.rnn_cell.GRUCell(self.emb_size)
            else:
                self.cell = tf.nn.rnn_cell.BasicLSTMCell(self.emb_size)
            if self.cell_type.endswith('/2'):
                self.cell = tf.nn.rnn_cell.MultiRNNCell([self.cell] * 2)

            self.output = self._rnn('rnn', self.inputs_ref, self.inputs_hyp,
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

        self.optimizer = tf.train.AdamOptimizer(self.alpha)
        self.train_func = self.optimizer.minimize(self.cost)

        # initialize session
        session_config = None
        if self.max_cores:
            session_config = tf.ConfigProto(inter_op_parallelism_threads=self.max_cores,
                                            intra_op_parallelism_threads=self.max_cores)
        self.session = tf.Session(config=session_config)

        # this helps us load/save the model
        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2)

    def _rnn(self, name, enc_inputs_ref, enc_inputs_hyp, enc_inputs_da=None):
        """Build the rating prediction RNN structure.
        @return: TensorFlow Output with the prediction
        """
        # TODO fix using word2vec
        # -- tady nejaky if s embedding_lookup; na zacatku musim mit variable, a muzu si rict, ze bude trainable=False
        with tf.variable_scope('enc_ref') as scope:
            enc_cell_ref = tf.nn.rnn_cell.EmbeddingWrapper(self.cell, self.dict_size, self.emb_size)
            enc_outs_ref, enc_state_ref = tf.nn.rnn(enc_cell_ref, enc_inputs_ref, dtype=tf.float32)
            if self.reuse_embeddings:
                scope.reuse_variables()
                enc_cell_hyp = tf.nn.rnn_cell.EmbeddingWrapper(self.cell, self.dict_size, self.emb_size)
                enc_outs_hyp, enc_state_hyp = tf.nn.rnn(enc_cell_hyp, enc_inputs_hyp, dtype=tf.float32)

        with tf.variable_scope('enc_hyp'):
            if not self.reuse_embeddings:
                enc_cell_hyp = tf.nn.rnn_cell.EmbeddingWrapper(self.cell, self.dict_size, self.emb_size)
                enc_outs_hyp, enc_state_hyp = tf.nn.rnn(enc_cell_hyp, enc_inputs_hyp, dtype=tf.float32)

        if enc_inputs_da is not None:
            with tf.variable_scope('enc_da'):
                enc_cell_da = tf.nn.rnn_cell.EmbeddingWrapper(self.cell, self.da_dict_size, self.emb_size)
                enc_outs_da, enc_state_da = tf.nn.rnn(enc_cell_da, enc_inputs_da, dtype=tf.float32)

        # concatenate last LSTM states & outputs (works for multilayer LSTMs&GRUs)
        last_outs_and_states = tf.concat(1, self._flatten_enc_state(enc_state_ref) +
                                         self._flatten_enc_state(enc_state_hyp) +
                                         (self._flatten_enc_state(enc_state_da)
                                          if enc_inputs_da is not None else []))
        state_size = int(last_outs_and_states.get_shape()[1])
        hidden = last_outs_and_states

        if self.tanh_layers > 0:
            with tf.variable_scope('hidden'):
                for layer_no in xrange(self.tanh_layers):
                    h_w = tf.get_variable(name + '-w' + str(layer_no + 1),
                                          (state_size, state_size),
                                          initializer=tf.random_normal_initializer(stddev=0.1))
                    h_b = tf.get_variable(name + '-b' + str(layer_no + 1),
                                          (state_size,),
                                          initializer=tf.constant_initializer())
                    hidden = tf.tanh(tf.matmul(hidden, h_w) + h_b)

        with tf.variable_scope('classif'):
            w = tf.get_variable(name + '-w', (state_size, self.num_outputs),
                                initializer=tf.random_normal_initializer(stddev=0.1))
            b = tf.get_variable(name + '-b', (self.num_outputs,),
                                initializer=tf.constant_initializer())

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

    def _add_inputs_to_feed_dict(self, fd, inputs_ref, inputs_hyp, inputs_da=None):

        fd[self.initial_state_ref] = np.zeros([inputs_ref.shape[0], self.emb_size])
        fd[self.initial_state_hyp] = np.zeros([inputs_hyp.shape[0], self.emb_size])

        # TODO check for None ??
        sliced_ref = np.squeeze(np.array(np.split(inputs_ref, len(inputs_ref[0]), axis=1)), axis=2)
        sliced_hyp = np.squeeze(np.array(np.split(inputs_hyp, len(inputs_hyp[0]), axis=1)), axis=2)

        for input_, slice_ in zip(self.inputs_ref, sliced_ref):
            fd[input_] = slice_
        for input_, slice_ in zip(self.inputs_hyp, sliced_hyp):
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

        pass_cost = 0

        for inst_nos in self._batches():

            log_debug('INST-NOS: ' + str(inst_nos))
            log_debug("\n".join(' '.join([tok for tok, _ in self.train_refs[i]]) + "\n" +
                                ' '.join([tok for tok, _ in self.train_hyps[i]]) + "\n" +
                                unicode(self.y[i])
                                for i in inst_nos))

            fd = {self.target: self.y[inst_nos]}
            self._add_inputs_to_feed_dict(fd, self.X_ref[inst_nos], self.X_hyp[inst_nos],
                                          self.X_da[inst_nos] if self.da_enc else None)
            results, cost, _ = self.session.run([self.output, self.cost, self.train_func],
                                                feed_dict=fd)
            log_debug('R: ' + str(results))
            log_debug('COST: %f' % cost)

            pass_cost += cost

        # print and return statistics
        self._print_pass_stats(pass_no, datetime.timedelta(seconds=(time.time() - pass_start_time)),
                               pass_cost)

        return pass_cost

    def _print_pass_stats(self, pass_no, time, cost):
        log_info('PASS %03d: duration %s, cost %f' % (pass_no, str(time), cost))

    def evaluate(self, inputs, raw_targets, output_file=None):
        """
        Evaluate the predictor on the given inputs & targets; possibly also write to a file.
        """
        if isinstance(inputs, tuple):
            das, input_refs, input_hyps = inputs
        else:
            das, input_refs, input_hyps = self._divide_inputs(inputs)
        dists = []
        correct = 0
        raw_ratings = []
        ratings = []
        targets = []
        for da, input_ref, input_hyp, target in zip(das, input_refs, input_hyps, raw_targets):
            rating = self.rate([input_ref], [input_hyp], [da] if self.da_enc else None)
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
                'accuracy': float(correct) / len(input_refs),
                'pearson': pearson,
                'pearson_pv': pearson_pv,
                'spearman': spearman,
                'spearman_pv': spearman_pv}
