# -*- coding: utf-8 -*-

from functools import reduce


def bpe2word(tokenized_sentence):
    func = lambda x, y: f"{x}{y.lstrip('##')}" if y.startswith('##') else f"{x} {y}"
    return reduce(func, tokenized_sentence).split(' ')


class ReplacingAugmentor(object):
    def __init__(self, tokenizer, selector, generator, to_word=False):
        self.tokenizer = tokenizer
        self.selector = selector
        self.generator = generator
        self.to_word = to_word

    def __call__(self, sentence, rate=0.1):
        words = self.tokenizer.tokenize(sentence)
        if self.to_word:
            words = bpe2word(words)
    
        selected_positions = self.selector(words, rate)
        sentence = ' '.join(self.generator(words, selected_positions))
        if self.to_word:
            sentence = ' '.join(self.tokenizer.tokenize(sentence))
        return sentence


# class SwapGenerator(object):
#     def __init__(self, windowsize=3):
#         self.ws = windowsize
# 
#     def __call__(self, words):
