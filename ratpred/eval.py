#!/usr/bin/env python
# coding=utf-8

from __future__ import unicode_literals

import scipy.stats
import numpy as np

from ratpred.futil import write_outputs, read_outputs


def stats_for_col(data, col_num, mask=None):
    """Transpose array of arrays + return a specific column (with original data grouped by rows)."""
    ret = np.array(data).transpose()[col_num]
    return ret[mask] if mask is not None else ret


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
        self.total = np.zeros((len(self.target_cols),))
        self.correct = np.zeros((len(self.target_cols),))
        self.rank_total = np.zeros((len(self.target_cols),))
        self.rank_ok = np.zeros((len(self.target_cols),))
        self.rank_loss = np.zeros((len(self.target_cols),))

    def append(self, inp, raw_trg, trg, raw_rat, rat, raw_rank_diff, rank_ok):
        """Append one rating (along with input & target), update statistics."""
        self.inputs.append(inp)
        self.raw_ratings.append(raw_rat)
        self.ratings.append(rat)
        self.raw_targets.append(raw_trg)
        self.targets.append(trg)
        # mask stuff that should not be evaluated
        aspect_mask = 1. - np.isnan(raw_trg)
        ranking_mask = int(inp[3] is not None) * np.ones(rank_ok.shape)
        ranking_mask *= aspect_mask
        aspect_mask *= 1. - ranking_mask
        # rating stats
        self.total += aspect_mask
        raw_trg = np.nan_to_num(raw_trg)
        trg = np.nan_to_num(raw_trg)
        self.dists.append(aspect_mask * abs(raw_rat - raw_trg))
        self.aes.append(aspect_mask * abs(rat - trg))
        self.sqes.append(aspect_mask * (rat - trg) ** 2)
        self.correct += aspect_mask * (trg == rat).astype(np.float)
        # ranking stats
        self.rank_total += ranking_mask
        self.rank_ok += ranking_mask * rank_ok
        self.rank_loss += ranking_mask * np.maximum(- raw_rank_diff, np.zeros_like(raw_rank_diff))

    def write_tsv(self, fname):
        """Write a TSV file containing all the prediction data so far."""
        outputs = {}
        for col_num, target_col in enumerate(self.target_cols):
            outputs[target_col] = {'human_rating_raw': stats_for_col(self.raw_targets, col_num),
                                   'human_rating': stats_for_col(self.targets, col_num),
                                   'system_rating_raw': stats_for_col(self.raw_ratings, col_num),
                                   'system_rating': stats_for_col(self.ratings, col_num),
                                   'rank_loss': stats_for_col(self.rank_loss, col_num),
                                   'rank_ok': stats_for_col(self.rank_ok, col_num)}
        write_outputs(fname, self.inputs, outputs)

    def get_stats(self):
        """Return important statistics (incl. correlations) in a dictionary."""
        ret = {}
        for col_num, target_col in enumerate(self.target_cols):
            mask = ~np.isnan(stats_for_col(self.raw_targets, col_num))
            rank_mask = np.array([inp[3] is not None for inp in self.inputs])
            rank_mask &= mask
            mask &= ~rank_mask
            pearson, pearson_pv = scipy.stats.pearsonr(stats_for_col(self.targets, col_num, mask),
                                                       stats_for_col(self.ratings, col_num, mask))
            spearman, spearman_pv = scipy.stats.spearmanr(stats_for_col(self.targets, col_num, mask),
                                                          stats_for_col(self.ratings, col_num, mask))
            ret[target_col] = {'dist_total': np.sum(stats_for_col(self.dists, col_num, mask)),
                               'dist_avg': np.mean(stats_for_col(self.dists, col_num, mask)),
                               'dist_stddev': np.std(stats_for_col(self.dists, col_num, mask)),
                               'mae': np.mean(stats_for_col(self.aes, col_num, mask)),
                               'rmse': np.sqrt(np.mean(stats_for_col(self.sqes, col_num, mask))),
                               'accuracy': self.correct[col_num] / self.total[col_num],
                               'rank_acc': self.rank_ok[col_num] / self.rank_total[col_num],
                               'rank_loss_total': self.rank_loss[col_num],
                               'rank_loss_avg': self.rank_loss[col_num] / self.rank_total[col_num],
                               'pearson': pearson,
                               'pearson_pv': pearson_pv,
                               'spearman': spearman,
                               'spearman_pv': spearman_pv}
        return ret

    def append_from_tsv(self, fname):
        inps, outs = read_outputs(fname)
        cols = sorted(outs.keys())
        # empty: initialize cols according to data
        if self.target_cols == [''] and not self.inputs:
            self.target_cols = cols
            self.reset()
        # we're non-empty: check compatibility
        else:
            assert self.target_cols == cols
        # convert format
        raw_trgs = np.array([outs[key]['human_rating_raw'] for key in cols]).transpose()
        trgs = np.array([outs[key]['human_rating'] for key in cols]).transpose()
        raw_rats = np.array([outs[key]['system_rating_raw'] for key in cols]).transpose()
        rats = np.array([outs[key]['system_rating'] for key in cols]).transpose()
        rank_losses = np.array([outs[key]['rank_loss'] for key in cols]).transpose()
        rank_oks = np.array([outs[key]['rank_ok'] for key in cols]).transpose()
        # append the instances
        for inp, raw_trg, trg, raw_rat, rat, rank_loss, rank_ok in zip(
                inps, raw_trgs, trgs, raw_rats, rats, rank_losses, rank_oks):
            self.append(inp, raw_trg, trg, raw_rat, rat, rank_loss, rank_ok)
