#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals


from argparse import ArgumentParser
import yaml
import codecs


def main(config_file, values):
    with codecs.open(config_file, 'r', 'UTF-8') as fh:
        cfg = yaml.load(fh)
    cfg.update(yaml.load(values))
    with codecs.open(config_file, 'w', 'UTF-8') as fh:
        yaml.dump(cfg, fh, default_flow_style=False, allow_unicode=True)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('config_file', type=str, help='Yaml config file, to be modified in-place')
    ap.add_argument('values', type=str, help='Yaml values to replace in the config file')
    args = ap.parse_args()
    main(args.config_file, args.values)
