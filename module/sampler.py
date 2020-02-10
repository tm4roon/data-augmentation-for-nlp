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
