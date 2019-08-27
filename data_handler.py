# -*- coding: utf-8 -*-

import math
import random

import torch
from functools import reduce


def data_loader(path, src_minlen, src_maxlen, tgt_minlen, tgt_maxlen, bos_token='[BOS]'):
    def preprocess(line):
        src, tgt = line.split('\t')
        src_words = src.rstrip().split(' ')
        tgt_words = [bos_token] + tgt.rstrip().split(' ')
        if src_minlen <= len(src_words) <= src_maxlen \
            and tgt_minlen <= len(tgt_words) <= tgt_maxlen:
            return (' '.join(src_words), ' '.join(tgt_words))

    with open(path) as f:
        data = [preprocess(line) for line in f]
        return data


class DataAugmentationIterator(object):
    def __init__(self, tokenizer, data, batchsize, augmentor=None, side='both',
        batch_first=False, shuffle=True, repeat=False):
        self.tokenizer = tokenizer
        self.pad_idx = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        self.mask_idx = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
        self.unk_idx = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

        self.data = data
        self.augmentor = augmentor
        self.side = side

        self.bsz = batchsize
        self.batch_first = batch_first

        self.shuffle = shuffle
        self.repeat = repeat

    def __len__(self):
        return math.ceil(len(self.data)/self.bsz)

    def __iter__(self):
        while True:
            self._init_batches()
            for batch in self.batches:
                yield batch
            if not self.repeat:
                return


    def _init_batches(self):
        # augment data and numericalize
        if self.augmentor is None:
            self.augmented_data = [self._numericalize(s) for s in self.data]
        else:
            self.augmented_data = [self._augment(s) for s in self.data]

        # shuffle the augumented data order
        if self.shuffle:
            self.augmented_data = random.sample(self.augmented_data, len(self.augmented_data))

        self.batches = [
            self._pad(self.augmented_data[i:i+self.bsz])
            for i in range(0, len(self.data), self.bsz)
        ]

    def _numericalize(self, sentences):
        if isinstance(sentences, str):
            return self.tokenizer.encode(sentences)
        return tuple(self._numericalize(s) for s in sentences)

    def _augment(self, sentence_pairs):
        if self.side == 'src':
            pair = (self.augmentor(sentence_pairs[0]), sentence_pairs[1])
        elif self.side == 'tgt': 
            pair = (sentence_pairs[0], self.augmentor(sentence_pairs[1]))
        elif self.side == 'both':
            pair =  tuple(self.augmentor(s) for s in sentence_pairs)
        else:
            raise NotImplementedError
        return self.numericalize(pair)

    def _pad(self, bs):
        def padding(bs):
            maxlen = max([len(b) for b in bs])
            return torch.tensor([b + [self.pad_idx for _ in range(maxlen-len(b))] for b in bs])

        srcs, tgts = zip(*bs)
        srcs = padding(srcs)
        tgts = padding(tgts)

        if not self.batch_first:
            srcs = srcs.t().contiguous()
            tgts = tgts.t().contiguous()
        return (srcs, tgts)

    def state_statics(self):
        def statics(data):
            tadd = lambda xs, ys: tuple(x+y for x, y in zip(xs, ys))
            counts = reduce(tadd, [(len(s), s.count(self.unk_idx)) for s in data])
            return {'n_tokens': counts[0], 'n_unks': counts[1]}
        srcs, tgts = zip(*self.data)
        return {'src': statics(self._numericalize(srcs)), 
                'tgt': statics(self._numericalize(tgts))}
