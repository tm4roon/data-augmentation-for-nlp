# -*- coding: utf-8 -*-

import math
import random

import torch
from functools import reduce
from tqdm import tqdm


class Dataset(object):
    def __init__(self, path, tokenizer, src_minlen, src_maxlen, tgt_minlen, 
        tgt_maxlen, bos_token='[BOS]', eos_token='[EOS]', separator='\t', test=False):
        self.tokenizer = tokenizer
        self.pad_idx = tokenizer.pad_token_id
        self.mask_idx = tokenizer.mask_token_id
        self.unk_idx = tokenizer.unk_token_id

        self.bos_token = bos_token
        self.eos_token = eos_token
        self.separator = separator

        self.src_minlen = src_minlen
        self.src_maxlen = src_maxlen
        self.tgt_minlen = tgt_minlen
        self.tgt_maxlen = tgt_maxlen

        self.test = test

        with open(path, 'r') as f:
            self.examples = [self.preprocess(line) for line in f]

    def __len__(self):
        return len(self.examples)

    def preprocess(self, line):
        pair = line.rstrip().split(self.separator)
        src_words = self.tokenizer.tokenize(pair[0])

        if not self.test:
            pair[1] = self.bos_token + ' ' + pair[1] + ' ' + self.eos_token
            tgt_words = self.tokenizer.tokenize(pair[1])
            if self.src_minlen <= len(src_words) <= self.src_maxlen \
            and self.tgt_minlen <= len(tgt_words) <= self.tgt_maxlen:
                return pair
        else:
            if self.src_minlen <= len(src_words) <= self.src_maxlen:
                return pair

    def state_statistics(self):
        def statistics(data):
            tadd = lambda xs, ys: tuple(x+y for x, y in zip(xs, ys))
            counts = reduce(tadd, [(len(s), s.count(self.unk_idx)) for s in data])
            return {'n_tokens': counts[0], 'n_unks': counts[1]}

        fields = zip(*self.examples)
        stats = [statistics(list(map(self.tokenizer.encode, field))) for field in fields]
        return {'src': stats[0], 'tgt': stats[1]} if not self.test else {'src': stats[0]}
    

class DataAugmentationIterator(object):
    def __init__(self, data, batchsize, side='both', augmentor=None, 
        batch_first=False, shuffle=True):
        self._initialize()
        self.data = data
        self.bsz = batchsize
        self.batch_first = batch_first
        self.shuffle = shuffle

        self.augmentor = augmentor
        self.side = side
        self.augmentation_rate = 0.0

    def __len__(self):
        return math.ceil(len(self.data)/self.bsz)

    def __iter__(self):
        while True:
            # self.augmentation_rate = self.scheduler(self.current_epoch)
            self._init_batches()
            for batch in self.batches:
                self.n_update += 1
                yield batch
            self.current_epoch += 1
            return
    
    def _initialize(self):
        self.n_update = 0
        self.current_epoch = 1

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
                pair[0] = self.augmentor(pair[0], self.augmentation_rate)
            if self.side in ['tgt', 'both']:
                input_sent = ' '.join(pair[1].split(' ')[1:-1])
                pair[1] = '[BOS] ' + self.augmentor(input_sent, self.augmentation_rate) + ' [EOS]'
        return self._numericalize(pair) 

    def _padding(self, bs):
        def pad(bs):
            maxlen = max([len(b) for b in bs])
            return torch.tensor([b + [self.data.pad_idx for _ in range(maxlen-len(b))] for b in bs])

        batches = zip(*bs)
        batches = [pad(batch) for batch in batches]

        if not self.batch_first:
            batches = [batch.t().contiguous() for batch in batches]
        return batches
