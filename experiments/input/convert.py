#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import pandas as pd
import numpy as np
import os.path
import re
import codecs
import pprint
from argparse import ArgumentParser

from tgen.data import DA
from tgen.delex import delex_sent
from futil.tokenize import tokenize
from tgen.logf import log_info
from tgen.lexicalize import Lexicalizer


# Start IPdb on error in interactive mode
from tgen.debug import exc_info_hook
import sys
sys.excepthook = exc_info_hook


DELEX_SLOTS = ['count', 'addr', 'area', 'food', 'price', 'phone',
               'near', 'pricerange', 'postcode', 'address', 'eattype',
               'type', 'price_range', 'good_for_meal', 'name']


def distort_sent(sent, distort_step, vocab):
    """Distort a sentence (add/remove/replace words).
    @param sent: the sentence to distort (list of tokens)
    @param distort_step: number of distortions to perform
    @param vocab: a tuple of (vocabulary list, corresponding unigram probabilities)
    @return: the distorted sentence, as a list of tokens
    """

    sent = list(sent)

    for i in xrange(distort_step):
        # select token to work on
        trials = 0
        tok_no = np.random.randint(0, len(sent))
        while trials < 100 and re.match(r'^([!?,:.]|the|an?)$', sent[tok_no]):
            tok_no = np.random.randint(0, len(sent))
            trials += 1
        tok = sent[tok_no]

        # select what to do (random distortion action)
        possible_actions = ['duplicate', 'replace', 'duplicate_rand', 'insert_rand']
        if len(sent) > 1:  # disallow removing the last word
            possible_actions.append('remove')
        action = np.random.choice(possible_actions)

        if action in ['insert_rand', 'replace']:  # generate random token to insert/replace
            trials = 0
            other_tok = np.random.choice(vocab[0], p=vocab[1])
            while trials < 100 and other_tok == tok:
                other_tok = np.random.choice(vocab[0], p=vocab[1])
                trials += 1

        # perform action
        if action == 'duplicate':  # duplicate token
            sent.insert(tok_no, tok)
        elif action == 'remove':  # remove token
            del sent[tok_no]
        elif action == 'replace':  # replace token by a random other token
            sent[tok_no] = other_tok
        elif action == 'insert_rand':
            sent.insert(tok_no, other_tok)
        elif action == 'duplicate_rand':  # duplicate token at random position
            sent.insert(np.random.randint(0, len(sent)), tok)

    return sent


def build_vocab(freq_dict):
    """Given a frequency dictionary collected on the training data, this builds a
    tuple of (lemmas, corresponding probabilities). Using +1 smoothing.
    @param freq_dict: a frequency dictionary collected from data (word -> # occurrences)
    @return: a tuple (word list, probability list)
    """

    norm_const = float(sum(freq_dict.values())) + len(freq_dict)  # +1 smoothing
    vocab_toks = []
    vocab_ps = []

    for tok, freq in freq_dict.iteritems():
        vocab_toks.append(tok)
        vocab_ps.append((freq + 1) / norm_const)  # +1 smoothing

    return vocab_toks, np.array(vocab_ps)


def create_fake_data(real_data, columns, score_type='nlg'):
    """Given some real data, create additional fake data, using human references and
    distorting them. Will start from scores provided, or default to best possible score.
    @param real_data: a real data set, as pd.DataFrame
    @param columns: list of columns for the fake data set
    @param score_type: switch between Likert scale 1-6 ('nlg') and HTER ('hter')
    @return: a fake data set, with the given columns, some of them empty
    """
    def target_score(src_score, distort_step):
        if score_type == 'hter':
            return src_score + distort_step
        return max(1, min(4., src_score - distort_step))

    normalize = False
    best_score = 6.
    num_steps = 4
    if score_type == 'hter':
        normalize = True
        best_score = 0.
        num_steps = 5

    fake_data = pd.DataFrame(index=np.arange(len(real_data) * (num_steps + 1)), columns=columns)
    vocab = {}

    # add references as perfect data items
    for idx, row in enumerate(real_data.itertuples()):
        fake_data.loc[idx]['orig_ref'] = row.orig_ref
        fake_data.loc[idx]['system_ref'] = row.orig_ref
        fake_data.loc[idx]['mr'] = row.mr
        fake_data.loc[idx]['is_real'] = 0
        for quant in ['naturalness', 'quality', 'informativeness']:
            fake_data.loc[idx][quant] = (getattr(row, quant)
                                         if (hasattr(row, quant) and
                                             getattr(row, quant) is not None and
                                             not np.isnan(getattr(row, quant)))
                                         else best_score)

        for tok in tokenize(row.orig_ref).split(' '):
            vocab[tok] = vocab.get(tok, 0) + 1

    lexicalizer = Lexicalizer(cfg={'mode': 'tokens'})  # default lexicalizer
    vocab = build_vocab(vocab)

    for distort_step in xrange(1, num_steps + 1):
        for idx, row in enumerate(real_data.itertuples(), start=distort_step * len(real_data)):

            fake_data.loc[idx]['orig_ref'] = row.orig_ref
            fake_data.loc[idx]['mr'] = row.mr
            fake_data.loc[idx]['is_real'] = 0

            # delexicalize data
            da = DA.parse_cambridge_da(row.mr)
            sent, _, lex_instr = delex_sent(da, tokenize(row.orig_ref).split(' '), DELEX_SLOTS)
            ref_len = len(sent)
            # distort
            sent = distort_sent(sent, distort_step, vocab)
            # lexicalize again
            sent = lexicalizer._tree_to_sentence([(tok, None) for tok in sent], lex_instr)
            fake_data.loc[idx]['system_ref'] = ' '.join(sent)

            for quant in ['naturalness', 'quality', 'informativeness']:
                score = (getattr(row, quant)
                         if (hasattr(row, quant) and
                             getattr(row, quant) is not None and
                             not np.isnan(getattr(row, quant)))
                         else best_score)
                score = target_score(score, distort_step)
                fake_data.loc[idx][quant] = (((score / ref_len) * 100) if normalize else score)

    return fake_data


def get_data_parts(data, sizes):
    """Split the data into parts, given the parts' proportional sizes.
    @param data: the data to be split
    @param sizes: parts' sizes ratios
    @return a list of data parts
    """
    parts = []
    total_parts = sum(sizes)
    sizes = [int(round((part / float(total_parts)) * len(data))) for part in sizes]
    # 1st part takes the rounding error
    sizes[0] += len(data) - sum(sizes)
    # make sure that there is no overlap in instances with identical (mr, system_ref)
    # this makes the set sizes not exactly equal, but should be OK
    offset = 0
    for i in xrange(len(sizes) - 1):
        while (sizes[i] < len(data) and
               all(data.ix[sizes[i] - 1, ['mr', 'system_ref']] ==
                   data.ix[sizes[i], ['mr', 'system_ref']])):
            sizes[i] += 1
            sizes[i + 1] -= 1
        offset += sizes[i]
    offset = 0
    for size in sizes:
        part = data.iloc[offset:offset + size, :]
        offset += size
        parts.append(part)
    return parts


def add_fake_data(train_data, real_data, add_from_train=False):
    if add_from_train:
        log_info("Will create fake data from system outputs in training data.")
        # we keep the scores here, but use the outputs as orig references
        sys_outs = train_data.copy()
        del sys_outs['orig_ref']  # delete original human refs first
        sys_outs = sys_outs.rename(columns={'system_ref': 'orig_ref'})
        real_data = pd.concat((real_data, sys_outs))

    # there is some fake data to be created and added
    if len(real_data):
        log_info("Creating fake data...")
        fake_data = create_fake_data(real_data, train_data.columns,
                                     score_type=('hter' if args.hter_score else 'nlg'))
        log_info("Created %d fake instances." % len(fake_data))
        return pd.concat([fake_data, train_data])

    # no fake data to be added -> return just the original
    return train_data


def convert(args):
    """Main function, does the conversion, taking parsed command-line arguments.
    @param args: command-line arguments
    """

    log_info("Loading %s..." % args.input_file)
    data = pd.read_csv(args.input_file, index_col=None,
                       sep=(b"\t" if args.input_file.endswith('.tsv') else b","),
                       encoding='utf-8')
    log_info("Contains %d instances." % len(data))

    # add dummy judge ids so that pandas does not drop stuff when groupping :-(
    if 'judge_id' in data.columns:
        data['judge_id'] = data['judge_id'].apply(lambda x: x if not np.isnan(x) else -1)
    # mark data as being "real" (as opposed to added fake data)
    data['is_real'] = pd.Series(np.ones(len(data), dtype=np.int32), index=data.index)

    fake_data_refs = pd.DataFrame(columns=['mr', 'orig_ref'])
    if args.create_fake_data_from:
        log_info("Will create fake data from %s." % args.create_fake_data_from)
        fake_data_refs = pd.concat((fake_data_refs,
                                    pd.read_csv(args.create_fake_data_from,
                                                index_col=None, sep=b"\t")))
    if 'H' in args.create_fake_data:
        log_info("Will create fake data from human refs in all data.")
        fake_data_refs = pd.concat((fake_data_refs,
                                    data.groupby(['mr', 'orig_ref'],  # delete scores
                                                 as_index=False).agg(lambda vals: None)))

    if args.delete_refs:
        log_info("Deleting human references...")
        group_cols = list(set(data.columns) - set(['orig_ref']))
        data = data.groupby(group_cols, as_index=False).agg(lambda vals: "")
        # TODO what?? fake_data['orig_ref'] = fake_data['orig_ref'].apply(lambda vals: "")

    if args.median:
        log_info("Computing medians...")
        group_cols = list(set(data.columns) - set(['informativeness', 'naturalness',
                                                  'quality', 'judge', 'judge_id']))
        data = data.groupby(group_cols, as_index=False).median()
    if args.concat_refs:
        log_info("Concatenating all references for the same outputs...")
        group_cols = list(set(data.columns) - set(['orig_ref']))
        data = data.groupby(group_cols, as_index=False).agg(lambda vals: ' <|> '.join(vals))

    # shuffle, but keep identical instances (just with different human refs) together
    if args.shuffle:
        log_info("Shuffling...")
        # set index for grouping (DAs should be disjunct across sets, so this should suffice)
        data = data.set_index(['mr', 'system_ref'])
        # do the actual shuffling over the new index
        data = data.loc[[tuple(p) for p in np.random.permutation(list(set(data.index)))]]
        # merge the index back into the dataset
        data = data.reset_index()
    sizes = [int(part) for part in args.ratio.split(':')]
    labels = args.labels.split(':')

    if args.devtest_crit:
        # select dev/test data based on a criterion
        crit_col, crit_val = args.devtest_crit.split('=')
        train_part = data[data[crit_col] != crit_val]  # training data is everything else
        train_part = train_part.reset_index()
        data = data[data[crit_col] == crit_val]  # dev+test data have the criterion
        data = data.reset_index()
        sizes = sizes[1:]  # training size does not matter (everything not fulfilling the criterion)
        if args.add_valid:  # add a few validation examples into training data
            train_part = pd.concat((train_part, data[0:args.add_valid]))
            data = data[args.add_valid:]

    if args.cv:  # for cross-validation, just pre-split the data to small parts (to be compiled later)
        cv_sizes = sizes
        sizes = [1] * sum(sizes)

    parts = get_data_parts(data, sizes)

    if args.devtest_crit:
        parts = [train_part] + parts

    if args.cv:  # for cross-validation, compile the data, repeating the parts with a shift
        cv_parts = []
        cv_labels = []
        for offset in xrange(len(sizes)):
            os.mkdir(os.path.join(args.output_dir, 'cv%02d' % offset))
            cur_parts = parts[offset:] + parts[:offset]
            cur_parts[0] = add_fake_data(cur_parts[0], fake_data_refs, 'S' in args.create_fake_data)
            for cv_size, cv_label in zip(cv_sizes, labels):
                cv_parts.append(pd.concat(cur_parts[:cv_size]))
                cur_parts = cur_parts[cv_size:]
                cv_labels.append(os.path.join('cv%02d' % offset, cv_label))
        labels = cv_labels
        parts = cv_parts
    else:
        parts[0] = add_fake_data(parts[0], fake_data_refs, 'S' in args.create_fake_data)

    if not os.path.isdir(args.output_dir):
        log_info("Directory %s not found, creating..." % args.output_dir)
        os.mkdir(args.output_dir)

    # mark down the configuration
    with codecs.open(os.path.join(args.output_dir, 'config'), 'wb', encoding='UTF-8') as fh:
        fh.write(pprint.pformat(vars(args), indent=4, width=100))

    for label, part in zip(labels, parts):
        # write the output
        log_info("Writing part %s (size %d)..." % (label, len(part)))
        part.to_csv(os.path.join(args.output_dir, label + '.tsv'),
                    sep=b"\t", index=False, encoding='utf-8', columns=sorted(part.columns))
    log_info("Done.")


if __name__ == '__main__':
    ap = ArgumentParser(description='Prepare and split data for the rating prediction experiment.')
    ap.add_argument('-d', '--devtest-crit', type=str, default=None,
                    help='A criterion (column=val) for selecting devel/test examples')
    ap.add_argument('-a', '--add-valid', type=int, default=0,
                    help='Add some validation data to train (use with --devtest-crit)')
    ap.add_argument('-r', '--ratio', type=str, default='3:1:1',
                    help='Train-devel-test split ratio')
    ap.add_argument('-l', '--labels', type=str, default='train:devel:test',
                    help='Train-devel-test labels')
    ap.add_argument('-s', '--shuffle', action='store_true',
                    help='Shuffle data before dividing?')
    ap.add_argument('-m', '--median', action='store_true',
                    help='Group human ratings and use medians')
    ap.add_argument('-D', '--delete-refs', action='store_true',
                    help='Remove human references (set field to "") and de-duplicate')
    ap.add_argument('-c', '--concat-refs', action='store_true',
                    help='Concatenate all human references and de-deduplicate')
    ap.add_argument('-v', '--cv', action='store_true',
                    help='Create cross-validation files (as many parts as ' +
                    'there are in the data split ratio)?')
    ap.add_argument('-F', '--create-fake-data-from', type=str,
                    help='Adding fake data from MRs + references in an additional file')
    ap.add_argument('-f', '--create-fake-data', type=str, default='',
                    help='Adding fake data from MRs + references in training data. ' +
                    '(values: H/S/HS, where H = human refs, S = system outputs)')
    ap.add_argument('-H', '--hter-score', action='store_true',
                    help='Use HTER score when generating the fake data, instead of NLG scores')
    ap.add_argument('input_file', type=str, help='Path to the input file')
    ap.add_argument('output_dir', type=str,
                    help='Output directory (where train,devel,test TSV will be created)')

    np.random.seed(1206)

    args = ap.parse_args()
    convert(args)
