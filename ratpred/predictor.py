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
import tempfile
import shutil

import numpy as np
import tensorflow as tf

from pytreex.core.util import file_stream

from tgen.logf import log_info, log_debug
from tgen.rnd import rnd
from tgen.embeddings import TokenEmbeddingSeq2SeqExtract
from tgen.tf_ml import TFModel

from ratpred.futil import read_data


class RatingPredictor(TFModel):

    def __init__(self, cfg):

        super(RatingPredictor, self).__init__(scope_name='predict_rating-' +
                                              cfg.get('scope_suffix', ''))
        self.cfg = cfg
        self.emb_size = cfg.get('emb_size', 50)
        self.embs = TokenEmbeddingSeq2SeqExtract(cfg)
        self.num_hidden_units = cfg.get('num_hidden_units', 512)

        self.passes = cfg.get('passes', 200)
        self.min_passes = cfg.get('min_passes', 0)
        self.alpha = cfg.get('alpha', 0.1)
        self.randomize = cfg.get('randomize', True)
        self.batch_size = cfg.get('batch_size', 1)

        self.validation_freq = cfg.get('validation_freq', 10)
        self.max_cores = cfg.get('max_cores')
        self.checkpoint_path = None

        self.target_col = cfg.get('target_col', 'quality')
        self.delex_slots = cfg.get('delex_slots', set())
        if self.delex_slots:
            self.delex_slots = set(self.delex_slots.split(','))
        self.reuse_embeddings = cfg.get('reuse_embeddings', False)
        self.tanh_layers = cfg.get('tanh_layers', 0)

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
        if self.checkpoint_path:
            shutil.copyfile(self.checkpoint_path, tf_session_fname)
        else:
            self.saver.save(self.session, tf_session_fname)

    def get_all_settings(self):
        """Get all settings except the trained model parameters (to be stored in a pickle)."""
        data = {'cfg': self.cfg,
                'embs': self.embs, }
        if self.embs:
            data['dict_size'] = self.dict_size
        return data

    def _save_checkpoint(self):
        """Save a checkpoint to a temporary path; set `self.checkpoint_path` to the path
        where it is saved; if called repeatedly, will always overwrite the last checkpoint."""
        if not self.checkpoint_path:
            fh, path = tempfile.mkstemp(".ckpt", "ratpred-", self.checkpoint_path)
            self.checkpoint_path = path
        log_info('Saving checkpoint to %s' % self.checkpoint_path)
        self.saver.save(self.session, self.checkpoint_path)

    def restore_checkpoint(self):
        if not self.checkpoint_path:
            return
        self.saver.restore(self.session, self.checkpoint_path)

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
        tf_session_fname = re.sub(r'(.pickle)?(.gz)?$', '.tfsess', model_fname)
        ret._init_neural_network()
        ret.saver.restore(ret.session, tf_session_fname)
        return ret

    def train(self, train_data_file, data_portion=1.0):
        """Run training on the given training data.
        """
        inputs, targets = read_data(train_data_file, self.target_col, self.delex_slots)

        log_info('Training rating predictor...')

        # initialize training
        self._init_training(inputs, targets, data_portion)

        # start training
        top_cost = float('nan')

        for iter_no in xrange(1, self.passes + 1):
            self.train_order = range(len(self.train_refs))
            if self.randomize:
                rnd.shuffle(self.train_order)
            pass_cost = self._training_pass(iter_no)

            if self.validation_freq and iter_no > self.min_passes and iter_no % self.validation_freq == 0:

                # TODO use validation data here
                comb_cost = pass_cost
                log_info('Cost: %8.3f' % comb_cost)

                # if we have the best model so far, save it as a checkpoint (overwrite previous)
                if math.isnan(top_cost) or comb_cost < top_cost:
                    top_cost = comb_cost
                    self._save_checkpoint()

        # restore last checkpoint (best performance on devel data)
        self.restore_checkpoint()

    def rate(self, refs, hyps):
        """
        """
        inputs_ref = np.array([self.embs.get_embeddings(sent) for sent in refs])
        inputs_hyp = np.array([self.embs.get_embeddings(sent) for sent in hyps])
        fd = {}
        self._add_inputs_to_feed_dict(inputs_ref, inputs_hyp, fd)
        return self.session.run(self.output, feed_dict=fd)

    def _init_training(self, inputs, targets, data_portion=1.0):
        """Initialize training.

        @param data_portion: portion of the training data to be used (0.0-1.0)
        """
        # store training data, make it smaller if necessary
        train_size = int(round(data_portion * len(inputs)))
        self.train_das = [da for da, _, _ in inputs[:train_size]]
        self.train_refs = [ref for _, ref, _ in inputs[:train_size]]
        self.train_hyps = [hyp for _, _, hyp in inputs[:train_size]]
        self.y = targets[:train_size]
        self.train_order = range(len(self.train_refs))
        log_info('Using %d training instances.' % train_size)

        # initialize input embeddings
        self.dict_size = self.embs.init_dict(self.train_refs)
        self.dict_size = self.embs.init_dict(self.train_hyps, dict_ord=self.dict_size)

        # convert training data to indexes
        self.X_ref = np.array([self.embs.get_embeddings(sent) for sent in self.train_refs])
        self.X_hyp = np.array([self.embs.get_embeddings(sent) for sent in self.train_hyps])

        # initialize I/O shapes
        self.input_shape = self.embs.get_embeddings_shape()

        # initialize NN classifier
        self._init_neural_network()
        # initialize the NN variables
        self.session.run(tf.initialize_all_variables())

    def _init_neural_network(self):
        """Create the neural network for classification"""

        # set TensorFlow random seed
        tf.set_random_seed(rnd.randint(-sys.maxint, sys.maxint))

        self.target = tf.placeholder(tf.float32, [None], name='target')

        with tf.variable_scope(self.scope_name):

            self.initial_state_ref = tf.placeholder(tf.float32, [None, self.emb_size])
            self.initial_state_hyp = tf.placeholder(tf.float32, [None, self.emb_size])
            self.inputs_ref = [tf.placeholder(tf.int32, [None], name=('enc_inp_ref-%d' % i))
                               for i in xrange(self.input_shape[0])]
            self.inputs_hyp = [tf.placeholder(tf.int32, [None], name=('enc_inp_hyp-%d' % i))
                               for i in xrange(self.input_shape[0])]
            self.cell = tf.nn.rnn_cell.BasicLSTMCell(self.emb_size)
            self.output = self._rnn('rnn', self.inputs_ref, self.inputs_hyp)

        # RSS cost (TODO divide by number of examples??)
        self.cost = tf.reduce_sum(tf.pow(self.target - self.output, 2))

        self.optimizer = tf.train.AdamOptimizer(self.alpha)
        self.train_func = self.optimizer.minimize(self.cost)

        # initialize session
        session_config = None
        if self.max_cores:
            session_config = tf.ConfigProto(inter_op_parallelism_threads=self.max_cores,
                                            intra_op_parallelism_threads=self.max_cores)
        self.session = tf.Session(config=session_config)

        # this helps us load/save the model
        self.saver = tf.train.Saver(tf.all_variables(), write_version=tf.train.SaverDef.V2)

        log_info("TF variables:\n" + "\n".join([v.name for v in tf.all_variables()]))

    def _rnn(self, name, enc_inputs_ref, enc_inputs_hyp):
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

        # LSTM state contains the output
        last_outs_and_states = tf.concat(1, [enc_state_ref.c, enc_state_ref.h,
                                             enc_state_hyp.c, enc_state_hyp.h])
        hidden = last_outs_and_states

        if self.tanh_layers > 0:
            with tf.variable_scope('hidden'):
                for layer_no in xrange(self.tanh_layers):
                    h_w = tf.get_variable(name + '-w' + str(layer_no + 1),
                                          ((self.cell.state_size.c + self.cell.state_size.h) * 2,
                                           (self.cell.state_size.c + self.cell.state_size.h) * 2),
                                          initializer=tf.random_normal_initializer(stddev=0.1))
                    h_b = tf.get_variable(name + '-b' + str(layer_no + 1),
                                          ((self.cell.state_size.c + self.cell.state_size.h) * 2,),
                                          initializer=tf.constant_initializer())
                    hidden = tf.tanh(tf.matmul(hidden, h_w) + h_b)

        with tf.variable_scope('classif'):
            w = tf.get_variable(name + '-w', ((self.cell.state_size.c + self.cell.state_size.h) * 2, 1),
                                initializer=tf.random_normal_initializer(stddev=0.1))
            b = tf.get_variable(name + '-b', (1,), initializer=tf.constant_initializer())

        return tf.matmul(hidden, w) + b

    def _batches(self):
        """Create batches from the input; use as iterator."""
        for i in xrange(0, len(self.train_order), self.batch_size):
            yield self.train_order[i: i + self.batch_size]

    def _add_inputs_to_feed_dict(self, inputs_ref, inputs_hyp, fd):

        fd[self.initial_state_ref] = np.zeros([inputs_ref.shape[0], self.emb_size])
        fd[self.initial_state_hyp] = np.zeros([inputs_hyp.shape[0], self.emb_size])

        # TODO check for None ??
        sliced_ref = np.squeeze(np.array(np.split(inputs_ref, len(inputs_ref[0]), axis=1)), axis=2)
        sliced_hyp = np.squeeze(np.array(np.split(inputs_hyp, len(inputs_hyp[0]), axis=1)), axis=2)

        for input_, slice_ in zip(self.inputs_ref, sliced_ref):
            fd[input_] = slice_
        for input_, slice_ in zip(self.inputs_hyp, sliced_hyp):
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
            self._add_inputs_to_feed_dict(self.X_ref[inst_nos], self.X_hyp[inst_nos], fd)
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

    def evaluate(self, inputs, targets):
        """
        """
        dist = 0.0
        for (da, input_ref, input_hyp), target in zip(inputs, targets):
            rating = self.rate([input_ref], [input_hyp])
            dist += abs(rating - target)
        return dist
