#!/usr/bin/env python
# -"- coding: utf-8 -"-

import numpy as np
import pandas as pd
from argparse import ArgumentParser

# Start IPdb on error in interactive mode
from tgen.debug import exc_info_hook
from ratpred.futil import read_data
import sys
sys.excepthook = exc_info_hook



def quantize(col, num_vals):
    step = 1 / float(num_vals)
    return [np.floor(val / step) for val in col]


def main(args):
    data = pd.DataFrame.from_csv(args.input_file, index_col=None, sep=b"\t", encoding="UTF-8")
    dd = read_data(args.input_file, 'quality', delex_slots=set('count,addr,area,food,price,phone,near,pricerange,postcode,address,eattype,type,price_range,good_for_meal,name'.split(',')), delex_das=True)
    data['mr_lex'] = data['mr']
    data['system_output'] = [' '.join([tok for tok, _ in inst[2]]) for inst in dd[0]]
    data['mr'] = [inst[0].to_cambridge_da_string() for inst in dd[0]]
    if args.average:
        avg_data = [pd.DataFrame.from_csv(fname, index_col=None, sep=b"\t", encoding="UTF-8")
                    for fname in args.average]
        avg_data = pd.concat(avg_data).reset_index(drop=True)
        data['average'] = np.mean(avg_data['quality'])
    else:
        data['average'] = np.mean(data['quality'])
    data = data[[args.metric, 'quality', 'mr', 'orig_ref', 'system_output']]
    data['human_rating'] = data['quality']
    del data['quality']
    data['human_rating_raw'] = data['human_rating']

    metric = np.array(data[args.metric])
    if args.normalize:
        lo = np.min(metric)
        hi = np.max(metric)
        if args.metric == 'average':  # wouldn't work for averages
            lo = 1.0
            hi = 6.0
        metric = (metric - lo) / (hi - lo)

    if args.quantize:
        qrange, qstep = args.quantize.split(':')
        qlo, qhi = qrange.split('-')
        qlo, qhi, qstep = float(qlo), float(qhi), float(qstep)
        num_vals = ((qhi - qlo) / qstep)
        metric = np.array(quantize(metric, num_vals))
        metric = qlo + metric * qstep

    data['system_rating_raw'] = metric
    data['system_rating'] = metric
    del data[args.metric]

    data.to_csv(args.output_file, sep=b"\t", index=False, encoding='UTF-8')


if __name__ == '__main__':
    ap = ArgumentParser(description='Convert metrics scores to 1-6 Likert scale '
                        + 'to compute correlations comparable to our system\'s predictions')
    ap.add_argument('--average', '-a', nargs='+', help='File to use to compute average metric')
    ap.add_argument('--normalize', '-n', action='store_true', help='Normalize metric to 0-1 range?')
    ap.add_argument('--quantize', '-q', default="1-6:0.5", help='Quantization parameters (range "1-6:0.5" step). Implies normalization!')
    ap.add_argument('metric', type=str, help='Metric name')
    ap.add_argument('input_file', type=str, help='Input TSV file with metrics')
    ap.add_argument('output_file', type=str, help='Output TSV file with metric as predicted rating')

    main(ap.parse_args())
