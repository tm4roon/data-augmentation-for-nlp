# -*- coding: utf-8 -*-

import argparse
import os
import dill

from tqdm import tqdm

import torch
from torchtext import data

from options import translate_opts
import utils

from models.transformer import Transformer


def main(args):
    device = torch.device('cuda' if args.gpu  else 'cpu')

    load_vars = torch.load(args.model)
    model_args = load_vars['args']
    model_params = load_vars['weights']

    dirname = os.path.dirname(args.model)
    SRC = utils.load_field(os.path.join(dirname, 'src.field'))
    TGT = utils.load_field(os.path.join(dirname, 'tgt.field'))
    fields = [('src', SRC), ('tgt', TGT)]

    with open(args.input, 'r') as f:
        examples = [data.Example.fromlist([line], [('src', SRC)]) for line in f]
    
    test_data = data.Dataset(examples, [('src', SRC)])
    test_iter = data.Iterator(
        test_data,
        batch_size=args.batch_size,
        train=False,
        shuffle=False,
        sort=False
    ) 

    model = Transformer(fields, model_args).to(device)
    model.load_state_dict(model_params)
    
    model.eval()
    for samples in tqdm(test_iter, total=len(test_iter)):
        srcs = samples.src.to(device)
        outs = model.generate(srcs, args.maxlen).transpose(0, 1)
        sents = [utils.id2w(out[1:], TGT) for out in outs]
        print('\n'.join(sents))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    translate_opts(parser)
    args = parser.parse_args()
    main(args)
