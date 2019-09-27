# -*- coding: utf-8 -*-

import random
import math
from abc import abstractmethod
from collections import Counter


class Sampler(object):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, words, rate):
        raise NotImplementedError


class UniformSampler(Sampler):
    def __call__(self, words, rate):
        rands = [i for i in range(len(words)) if random.uniform(0.0, 1.0) < rate]
        return rands


class AbsDiscountSampler(Sampler):
    def __init__(self, path):
        with open(path, 'r') as f:
            self.uniq = Counter()
            self.freq = Counter()
            for line in f:
                line = line.rstrip().split('\t')
                bigram = line[0].split(' ')
                freq = int(line[1])
                self.uniq[bigram[0]] += 1
                self.freq[bigram[0]] += freq

    def __call__(self, words, rate):
        return [i for i in range(len(words)) 
            if random.uniform(0.0, 1.0) < self._get_rate(words[i], rate)]

    def _get_rate(self, word, rate):
            try:
                return rate * self.uniq[word] / self.freq[word]
            except ZeroDivisionError:
                return 1.0
 
