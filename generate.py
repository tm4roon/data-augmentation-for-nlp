# -*- coding: utf-8 -*-

import argparse
import json
import os

from pytorch_transformers import BertTokenizer as Tokenizer

import options

from augmentation_module.dataset import (
    Dataset,
    DataAugmentationIterator,
)

from augmentation_module import augmentor
from augmentation_module import generator
from augmentation_module import sampler
from augmentation_module import scheduler
 

def main(args):

    # set tokenizer
    tokenizer = Tokenizer(
        args.vocab_file,
        do_basic_tokenize=False, 
        bos_token='[BOS]',
        eos_token='[EOS]',
        unk_token='[UNK]',
        sep_token='[SEP]',
        pad_token='[PAD]',
    )

    # load dataset for training and validation
    data = Dataset(
        args.input, tokenizer, 
        args.src_minlen, args.src_maxlen,
        args.tgt_minlen, args.tgt_maxlen, 
    )

    # set data-augmentation module
    to_word = False

    if args.sampling_strategy == 'random':
        sampling_fn = sampler.UniformSampler()
    elif args.sampling_strategy == 'absolute_discounting':
        sampling_fn = sampler.AbsDiscountSampler(
        args.bigram_frequency_for_sampling)
        to_word = True

    if args.augmentation_strategy == 'base':
        augmentor_fn = None
    else:
        if args.augmentation_strategy == 'dropout':
            generator_fn = generator.DropoutGenerator()
        elif args.augmentation_strategy == 'blank':
            generator_fn = generator.BlankGenerator(
                mask_token=tokenizer.mask_token)
        elif args.augmentation_strategy == 'unigram':
            generator_fn = generator.UnigramGenerator(
                args.unigram_frequency_for_generation)
            to_word=True
        elif args.augmentation_strategy == 'bigramkn':
            generator_fn = generator.BigramKNGenerator(
                args.bigram_frequency_for_generation)
            to_word=True
        elif args.augmentation_strategy == 'wordnet':
            generator_fn = generator.WordNetGenerator(lang='jpn')
            to_word = True
        elif args.augmentation_strategy == 'word2vec':
            generator_fn = generator.Word2vecGenerator(args.w2v_file)
            to_word = True
        elif args.augmentation_strategy == 'ppdb':
            generator_fn = generator.PPDBGenerator(args.ppdb_file)
            to_word = True
        elif args.augmentation_strategy == 'bert':
            generator_fn = generator.BertGenerator()
        augmentor_fn = augmentor.ReplacingAugmentor(
            tokenizer, sampling_fn, generator_fn, to_word=to_word)

    iterator = DataAugmentationIterator(
        data=data,
        batchsize=1,
        side=args.side,
        augmentor=augmentor_fn,
        shuffle=False,
    )

    scheduler.ConstantAR(
        iterator=iterator, 
        augmentation_rate=args.augmentation_rate,
    )

    iterator._init_batches()

    for pair in iterator.augmented_data:
        src = tokenizer.decode(pair[0])
        tgt = tokenizer.decode(pair[1][1:-1])
        print(src + '\t' + tgt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    options.generate_opts(parser)
    options.sub_opts(parser)
    args = parser.parse_args()
    main(args)
