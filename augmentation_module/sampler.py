# -*- coding: utf-8 -*-

import random
import math
from abc import abstractmethod


class Sampler(object):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, words, rate):
        raise NotImplementedError


class RandomSampler(Sampler):
    def __call__(self, words, rate):
        return random.sample(list(range(len(words))), math.floor(len(words)*rate))
