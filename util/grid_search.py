#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals

from argparse import ArgumentParser
import os
import yaml
import sys
from copy import copy


def main(one_dim, values, args):
    # grid-expand the different variable values
    cfg_overrides = [{}]
    for var in values:
        var = yaml.load(var)
        var_name = var.keys()[0]
        var_vals = var.values()[0]
        new_vals = []
        if one_dim:
            for cur_val in cfg_overrides[1:]:
                new_val = copy(cur_val)
                new_val[var_name] = var_vals[0]
                new_vals.append(new_val)
        for var_val in var_vals:
            cur_vals = [cfg_overrides[0]] if one_dim else cfg_overrides
            for cur_val in cur_vals:
                new_val = copy(cur_val)
                new_val[var_name] = var_val
                new_vals.append(new_val)
        cfg_overrides = new_vals

    # run the makefiles
    args = ('"' if args else '') + '" "'.join(args) + ('"' if args else '')
    for cfg_override in cfg_overrides:
        command = 'make cv_run %s CFG_OVERRIDE="%s"' % (args, yaml.dump(cfg_override)[:-1])
        print('Running: ' + command, file=sys.stderr)
        os.system(command)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('-1', '--one-dim', '--onedim', action='store_true',
                    help='Search along one dimension at a time (1st value=default)?')
    ap.add_argument('-v', '--values', nargs='+', type=str,
                    help='YAML-encoded lists of values to try out')
    ap.add_argument('args', nargs='*', type=str,
                    help='Args to pass through to CV run')
    args = ap.parse_args()
    main(args.one_dim, args.values, args.args)
