#!/usr/bin/env python
# coding=utf-8

from __future__ import unicode_literals

import scipy.stats
import numpy as np

from ratpred.futil import write_outputs, read_outputs


def stats_for_col(data, col_num):
    """Transpose array of arrays + return a specific column (with original data grouped by rows)."""
    return np.array(data).transpose()[col_num]


class Evaluator(object):
    """A cumulative statistics container to store predictions and later compute correlations."""

    def __init__(self, target_cols=None):
        self.target_cols = target_cols if target_cols else ['']
        self.reset()

    def reset(self):
        """Zero out all current statistics, start from scratch."""
        self.inputs = []
        self.raw_targets = []
        self.targets = []
        self.raw_ratings = []
        self.ratings = []
        self.dists = []
        self.aes = []  # absolute error
        self.sqes = []  # square error
        self.correct = np.zeros((len(self.target_cols),))

    def append(self, inp, raw_trg, trg, raw_rat, rat):
        """Append one rating (along with input & target), update statistics."""
        self.inputs.append(inp)
        self.raw_ratings.append(raw_rat)
        self.ratings.append(rat)
        self.raw_targets.append(raw_trg)
        self.targets.append(trg)
        self.dists.append(abs(raw_rat - raw_trg))
        self.aes.append(abs(rat - trg))
        self.sqes.append((rat - trg) ** 2)
        self.correct += (trg == rat).astype(np.float)

    def write_tsv(self, fname):
        """Write a TSV file containing all the prediction data so far."""
        outputs = {}
        for col_num, target_col in enumerate(self.target_cols):
            outputs[target_col] = {'human_rating_raw': stats_for_col(self.raw_targets, col_num),
                                   'human_rating': stats_for_col(self.targets, col_num),
                                   'system_rating_raw': stats_for_col(self.raw_ratings, col_num),
                                   'system_rating': stats_for_col(self.ratings, col_num)}
        write_outputs(fname, self.inputs, outputs)

    def get_stats(self):
        """Return important statistics (incl. correlations) in a dictionary."""
        ret = {}
        for col_num, target_col in enumerate(self.target_cols):
            pearson, pearson_pv = scipy.stats.pearsonr(stats_for_col(self.targets, col_num),
                                                       stats_for_col(self.ratings, col_num))
            spearman, spearman_pv = scipy.stats.spearmanr(stats_for_col(self.targets, col_num),
                                                          stats_for_col(self.ratings, col_num))
            ret[target_col] = {'dist_total': np.sum(stats_for_col(self.dists, col_num)),
                               'dist_avg': np.mean(stats_for_col(self.dists, col_num)),
                               'dist_stddev': np.std(stats_for_col(self.dists, col_num)),
                               'mae': np.mean(stats_for_col(self.aes, col_num)),
                               'rmse': np.sqrt(np.mean(stats_for_col(self.sqes, col_num))),
                               'accuracy': float(self.correct[col_num]) / len(self.inputs),
                               'pearson': pearson,
                               'pearson_pv': pearson_pv,
                               'spearman': spearman,
                               'spearman_pv': spearman_pv}
        return ret

    def append_from_tsv(self, fname):
        data = read_outputs(fname)
        for inp, raw_trg, trg, raw_rat, rat in zip(*data):
            self.append(inp, raw_trg, trg, raw_rat, rat)
