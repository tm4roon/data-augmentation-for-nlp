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
        unk_token='[UNK]',
        sep_token='[SEP]',
        pad_token='[PAD]',
        mask_token='[MASK]',
        cls_token='[CLS]',
    )

    tokenizer = WordpieceTokenizer(vocab)

    to_word = False

    # select a sampling module
    if args.sampling_strategy == 'random':
        sampling_fn = sampler.UniformSampler()

    # select a augmentation module
    if args.augmentation_strategy == 'dropout':
        generator_fn = generator.DropoutGenerator()
    elif args.augmentation_strategy == 'blank':
        generator_fn = generator.BlankGenerator(
            mask_token=tokenizer.vocab.mask_token)
    elif args.augmentation_strategy == 'unigram':
        generator_fn = generator.UnigramGenerator(
            args.unigram_frequency_for_generation)
        to_word=True
    elif args.augmentation_strategy == 'bigramkn':
        generator_fn = generator.BigramKNGenerator(
            args.bigram_frequency_for_generation)
        to_word=True
    elif args.augmentation_strategy == 'wordnet':
        generator_fn = generator.WordNetGenerator(lang=args.lang_for_wordnet)
        to_word = True
    elif args.augmentation_strategy == 'word2vec':
        generator_fn = generator.Word2vecGenerator(args.w2v_file)
        to_word = True
    elif args.augmentation_strategy == 'ppdb':
        generator_fn = generator.PPDBGenerator(args.ppdb_file)
        to_word = True
    elif args.augmentation_strategy == 'bert':
        from pytorch_transformers import BertTokenizer, BertForMaskedLM
        bert = BertForMaskedLM.from_pretrained(args.model_name_or_path)
        generator_fn = generator.BertGenerator(tokenizer, bert, args.temparature)

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
