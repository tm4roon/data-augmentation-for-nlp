# -*- coding: utf-8 -*-

import math
import random
import copy

import torch
from functools import reduce
from tqdm import tqdm


from .import utils


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
        self.examples = list(filter(lambda x: x is not None, self.examples))

    def __len__(self):
        return len(self.examples)

    def preprocess(self, line):
        pair = line.rstrip().split(self.separator)
        src_words = self.tokenizer.tokenize(utils.normalize(pair[0]))

        if not self.test:
            normalized = utils.normalize(pair[1])
            pair[1] = f'{self.bos_token} {normalized} {self.eos_token}'
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
        stats = [statistics(list(map(self.tokenizer.encode, field))) 
                    for field in fields]
        return {'src': stats[0], 'tgt': stats[1]} \
                if not self.test else {'src': stats[0]}
    

class DataAugmentationIterator(object):
    def __init__(self, data, batchsize, init_rate, side='src', augmentor=None, 
        batch_first=False, shuffle=True):
        self._initialize()
        self.data = data
        self.src_minlen = data.src_minlen
        self.src_maxlen = data.src_maxlen
        self.tgt_minlen = data.tgt_minlen
        self.tgt_maxlen = data.tgt_maxlen
        self.bsz = batchsize
        self.batch_first = batch_first
        self.shuffle = shuffle

        self.augmentor = augmentor
        self.side = side
        self.augmentation_rate = init_rate
        self._init_batches()

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        while True:
            for batch in self.batches:
                self.n_update += 1
                yield batch
            self.current_epoch += 1
            self._init_batches()
            return
    
    def _initialize(self):
        self.n_update = 0
        self.current_epoch = 1

    def _init_batches(self):
        # augment data and numericalize
        augmented_data = [self._augment(s) for s in self.data.examples]
        augmented_data = list(filter(lambda x: x is not None, augmented_data))

        # shuffle the data order
        if self.shuffle:
            augmented_data = random.sample(augmented_data, len(augmented_data))

        self.batches = [
            self._padding(augmented_data[i:i+self.bsz])
            for i in range(0, len(augmented_data), self.bsz)
        ]

    def _numericalize(self, sentences):
        if isinstance(sentences, str):
            return self.data.tokenizer.encode(sentences)
        return tuple(self._numericalize(s) for s in sentences)

    def _augment(self, pair):
        augmented_pair = copy.deepcopy(pair)
        if self.augmentor is not None:
            if self.side in ['src', 'both']:
                augmented_pair[0] = self.augmentor(pair[0], self.augmentation_rate) 
            if self.side in ['tgt', 'both']:
                input_sent = ' '.join(pair[1].split(' ')[1:-1])
                augmented_pair[1] = self.augmentor(input_sent, self.augmentation_rate)
                augmented_pair[1] = '[BOS] ' + augmented_pair[1] + ' [EOS]'
        numericalized_pair = self._numericalize(augmented_pair)

        if len(numericalized_pair) == 1: # test
            if self.src_minlen <= len(numericalized_pair[0]) <= self.src_maxlen:
                return numericalized_pair
        else: # train 
            if self.src_minlen <= len(numericalized_pair[0]) <= self.src_maxlen \
                and self.tgt_minlen <= len(numericalized_pair[1]) <= self.tgt_maxlen:
                return numericalized_pair

    def _padding(self, bs):
        def pad(bs):
            l = max([len(b) for b in bs])
            return torch.tensor([b + [self.data.pad_idx for _ in range(l-len(b))] for b in bs])

        batches = zip(*bs)
        batches = [pad(batch) for batch in batches]

        if not self.batch_first:
            batches = [batch.t().contiguous() for batch in batches]
        return batches
