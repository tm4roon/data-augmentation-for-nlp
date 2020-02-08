# -*- coding: utf-8 -*-

import argparse
import json
import os

import options

from module import augmentor
from module import generator
from module import sampler

from utils.tokenizer import PreDefinedVocab
from utils.tokenizer import WordpieceTokenizer


def main(args):

    # set tokenizer
    vocab = PreDefinedVocab(
        vocab_file=args.vocab_file,
        bos_token='[BOS]',
        eos_token='[EOS]',
        unk_token='[UNK]',
        sep_token='[SEP]',
        pad_token='[PAD]',
    )

    tokenizer = WordpieceTokenizer(vocab)

    to_word = False

    # select a sampling module
    if args.sampling_strategy == 'random':
        sampling_fn = sampler.UniformSampler()
    elif args.sampling_strategy == 'absolute_discounting':
        sampling_fn = sampler.AbsDiscountSampler(
        args.bigram_frequency_for_sampling)
        to_word = True

    # select a augmentation module
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

    with open(args.input, 'r') as f:
        for line in f:
            line = line.rstrip()
            augmented_line = augmentor_fn(line, args.augmentation_rate)
            print(augmented_line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    options.generate_opts(parser)
    options.sub_opts(parser)
    args = parser.parse_args()
    main(args)
