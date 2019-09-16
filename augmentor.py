# -*- coding: utf-8 -*-


import random
import math


def bpe2unidic(tokenized_sentence):
    func = lambda x, y: f'{x}{y}' if y.startswith('##') else f'{x} {y}'
    return reduce(func, tokenized_sentence).split(' ')

        

class ReplacingAugmentor(object):
    def __init__(self, tokenizer, selector, generator):
        self.tokenizer = tokenizer
        self.selector = selector
        self.generator = generator

    def __call__(self, sentence, rate=0.1, to_unidic=True):
        words = self.tokenizer.tokenize(sentence)
        if to_unidic:
            words = bpe2unidic(words)
        import pdb; pdb.set_trace()
        selected_positions = self.selector(words, rate)
        return self.generator(words, selected_position)


class RandomSelector(object):
    def __init__(self):
        pass

    def __call__(self, words, rate):
        return random.sample(words, ceil(len(words)*rate))


# class SwapGenerator(object):
#     def __init__(self, windowsize=3):
#         self.ws = windowsize
# 
#     def __call__(self, words, positions):
    
        
class DropGenerator(object):


# class BlankGenerator(object):


# class SmoothGenerator(object):


# class WordNetGenerator(object):


# class BERTGenerator(object):

