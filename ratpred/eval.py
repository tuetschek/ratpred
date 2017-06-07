#!/usr/bin/env python
# coding=utf-8

from __future__ import unicode_literals

import scipy.stats
import numpy as np

from ratpred.futil import write_outputs, read_outputs


class Evaluator(object):
    """A cumulative statistics container to store predictions and later compute correlations."""

    def __init__(self):
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
        self.correct = 0

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
        self.correct += 1 if trg == rat else 0

    def write_tsv(self, fname):
        """Write a TSV file containing all the prediction data so far."""
        write_outputs(fname, self.inputs,
                      self.raw_targets, self.targets, self.raw_ratings, self.ratings)

    def get_stats(self):
        """Return important statistics (incl. correlations) in a dictionary."""
        pearson, pearson_pv = scipy.stats.pearsonr(self.targets, self.ratings)
        spearman, spearman_pv = scipy.stats.spearmanr(self.targets, self.ratings)
        return {'dist_total': np.sum(self.dists),
                'dist_avg': np.mean(self.dists),
                'dist_stddev': np.std(self.dists),
                'mae': np.mean(self.aes),
                'rmse': np.sqrt(np.mean(self.sqes)),
                'accuracy': float(self.correct) / len(self.inputs),
                'pearson': pearson,
                'pearson_pv': pearson_pv,
                'spearman': spearman,
                'spearman_pv': spearman_pv}

    def append_from_tsv(self, fname):
        data = read_outputs(fname)
        for inp, raw_trg, trg, raw_rat, rat in zip(*data):
            self.append(inp, raw_trg, trg, raw_rat, rat)
