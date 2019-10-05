# -*- coding: utf-8 -*-

import argparse
import os
import dill

from tqdm import tqdm

import torch

from pytorch_transformers import BertTokenizer as Tokenizer

import options

from augmentation_module.dataset import (
    Dataset,
    DataAugmentationIterator,
)

from models.encoder import TransformerEncoder
from models.decoder import TransformerDecoder
from models.transformer import (
    Transformer,
    TranslationLM,
)

def id2w(tokenizer, pred):
    words = tokenizer.decode(pred).split(' ')
    if '[EOS]' in words:
        words = words[1:words.index('[EOS]')]
    return ' '.join(words)


def main(args):
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available else 'cpu')

    loaddir = os.path.dirname(args.model)

    tokenizer = Tokenizer(
        os.path.join(loaddir, 'vocab.txt'),
        do_basic_tokenize=False, 
        bos_token='[BOS]',
        eos_token='[EOS]',
        unk_token='[UNK]',
        sep_token='[SEP]',
        pad_token='[PAD]',
    )

    load_vars = torch.load(args.model)
    train_args = load_vars['args']
    model_params = load_vars['weights']

    test_data = Dataset(
        args.input, tokenizer,
        train_args.src_minlen, train_args.src_maxlen,
        train_args.tgt_minlen, train_args.tgt_maxlen,
        test=True,
    )

    test_iter = DataAugmentationIterator(
        data=test_data,
        batchsize=args.batchsize,
        init_rate=0.0,
        augmentor=None,
        shuffle=False,
    )

    pad_idx = tokenizer.pad_token_id
    bos_idx = tokenizer.bos_token_id

    encoder = TransformerEncoder(train_args, len(tokenizer), pad_idx)
    decoder = TransformerDecoder(train_args, len(tokenizer), pad_idx)
    model = Transformer(encoder, decoder, bos_idx).to(device)
    # elif train_args.arch == 'translm':
    #     sep_idx = tokenizer.sep_token_id
    #     model = TranslationLM(train_args, len(tokenizer), 
    #         pad_idx, bos_idx, sep_idx).to(device)
    model.load_state_dict(model_params)

    model.eval()
    pbar = tqdm(test_iter, dynamic_ncols=True)
    for samples in pbar:
        pbar.set_description(f'[translate]')
        srcs = samples[0].to(device)
        outs = model.generate(srcs, args.maxlen).transpose(0, 1).tolist()
        sents = [id2w(tokenizer, out) for out in outs]
        print('\n'.join(sents))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    options.translate_opts(parser)
    args = parser.parse_args()
    main(args)
