#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import unicode_literals

import tensorflow as tf
import os


class DummyTensorBoardLogger(object):

    def __init__(self):
        pass

    def add_to_log(self, cur_step, var_values):
        pass

    def create_tensor_summaries(self, tensor):
        pass

    def get_merged_summaries(self):
        return tf.constant(0.0, dtype=tf.float32, name='dummy_summary')

    def log(self):
        pass

class TensorBoardLogger(DummyTensorBoardLogger):

    def __init__(self, log_dir, run_id):
        self.log_dir = log_dir
        self.run_id = run_id
        self.tb_writer = tf.summary.FileWriter(os.path.join(self.log_dir, self.run_id))

    def add_to_log(self, cur_step, var_values):

        with tf.variable_scope('tb_logger'):
            val_list = tf.Summary(
                value=[tf.Summary.Value(tag=name, simple_value=float(value))
                       for name, value in var_values.iteritems()])

        self.tb_writer.add_summary(val_list, cur_step)

    def create_tensor_summaries(self, tensor):
        with tf.name_scope('summaries'):
          mean = tf.reduce_mean(tensor)
          tf.summary.scalar('mean', mean)
          with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(tensor - mean)))
          tf.summary.scalar('stddev', stddev)
          tf.summary.scalar('max', tf.reduce_max(tensor))
          tf.summary.scalar('min', tf.reduce_min(tensor))
          tf.summary.histogram('histogram', tensor)

    def get_merged_summaries(self):
        return tf.summary.merge_all()

    def log(self, cur_step, merged_summaries):
        self.tb_writer.add_summary(merged_summaries, cur_step)


