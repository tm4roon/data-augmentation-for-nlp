# -*- coding: utf-8 -*-

from abc import abstractmethod


class Generator(object):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, words, positions):
        raise NotImplementedError


class DropoutGenerator(Generator):
    def __call__(self, words, positions):
        return [words[i] for i in range(len(words)) if i not in positions]


class BlankGenerator(Generator):
    def __init__(self, mask_token='[MASK]'):
        self.mask_token = mask_token

    def __call__(self, words, positions):
        for i in positions:
            words[i] = self.mask_token
        return words

# [TODO]
class SmoothGenerator(object):
    def __init__(self):
        raise NotImplementedError


# [TODO]
class WordNetGenerator(object):
    def __init__(self):
        raise NotImplementedError


# [TODO]
class Word2vecGenerator(object):
    def __init__(self):
        raise NotImplementedError


# [TODO]
class BERTGenerator(object):
    def __init__(self):
        raise NotImplementedError
