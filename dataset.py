# -*- coding: utf-8 -*-

import math
import random

import torch
from functools import reduce


class Dataset(object):
    def __init__(self, path, tokenizer, src_minlen, src_maxlen, 
        tgt_minlen, tgt_maxlen, bos_token='[BOS]'):
        self.tokenizer = tokenizer
        self.pad_idx = tokenizer.pad_token_id
        self.mask_idx = tokenizer.mask_token_id
        self.unk_idx = tokenizer.unk_token_id

        self.src_minlen = src_minlen
        self.src_maxlen = src_maxlen
        self.tgt_minlen = tgt_minlen
        self.tgt_maxlen = tgt_maxlen
        self.bos_token = bos_token

        with open(path, 'r') as f:
            self.examples = [self.preprocess(line) for line in f]

    def __len__(self):
        return len(self.examples)

    def preprocess(self, line):
        src, tgt = line.rstrip().split('\t')
        src_words = self.tokenizer.tokenize(src)
        tgt_words = [self.bos_token] + self.tokenizer.tokenize(tgt)

        if self.src_minlen <= len(src_words) <= self.src_maxlen \
            and self.tgt_minlen <= len(tgt_words) <= self.tgt_maxlen:
            return (src, tgt)

    def state_statics(self):
        def statics(data):
            tadd = lambda xs, ys: tuple(x+y for x, y in zip(xs, ys))
            counts = reduce(tadd, [(len(s), s.count(self.unk_idx)) for s in data])
            return {'n_tokens': counts[0], 'n_unks': counts[1]}
        srcs, tgts = zip(*self.examples)
        return {'src': statics(list(map(self.tokenizer.encode, srcs))), 
                'tgt': statics(list(map(self.tokenizer.encode, tgts)))}

# def load(path, src_minlen, src_maxlen, tgt_minlen, tgt_maxlen, bos_token='[BOS]'):
#     def preprocess(line):
#         src, tgt = line.split('\t')
#         src_words = src.rstrip().split(' ')
#         tgt_words = [bos_token] + tgt.rstrip().split(' ')
#         if src_minlen <= len(src_words) <= src_maxlen \
#             and tgt_minlen <= len(tgt_words) <= tgt_maxlen:
#             return (' '.join(src_words), ' '.join(tgt_words))
# 
#     with open(path) as f:
#         data = [preprocess(line) for line in f]
#         return data


class DataAugmentationIterator(object):
    def __init__(self, data, batchsize, augmentor=None, side='both',
        batch_first=False, shuffle=True, repeat=False):

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
        self.augmented_data = [self._augment(s) for s in self.data.examples]

        # shuffle the augumented data order
        if self.shuffle:
            self.augmented_data = random.sample(self.augmented_data, len(self.augmented_data))

        self.batches = [
            self._padding(self.augmented_data[i:i+self.bsz])
            for i in range(0, len(self.data), self.bsz)
        ]

    def _numericalize(self, sentences):
        if isinstance(sentences, str):
            return self.data.tokenizer.encode(sentences)
        return tuple(self._numericalize(s) for s in sentences)

    def _augment(self, pair):
        if self.augmentor is not None:
            if self.side in ['src', 'both']:
                pair[0] = self.augmentor(pair[0])
            if self.side in ['tgt', 'both']:
                pair[1] = self.augmentor(pair[1])
        return self._numericalize(pair) 

    def _padding(self, bs):
        def pad(bs):
            maxlen = max([len(b) for b in bs])
            return torch.tensor([b + [self.data.pad_idx for _ in range(maxlen-len(b))] for b in bs])

        srcs, tgts = zip(*bs)
        srcs = pad(srcs)
        tgts = pad(tgts)

        if not self.batch_first:
            srcs = srcs.t().contiguous()
            tgts = tgts.t().contiguous()
        return (srcs, tgts)

