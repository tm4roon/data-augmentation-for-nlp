# -*- coding: utf-8 -*-

import random
import copy
import os

import numpy as np

from functools import reduce
from operator import concat
from abc import abstractmethod
from collections import Counter

import torch


def bpe2word(tokenized_sentence):
    func = lambda x, y: f"{x}{y.lstrip('##')}" \
        if y.startswith('##') else f"{x} {y}"
    return reduce(func, tokenized_sentence)


class BaseGenerator(object):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, words, positions):
        raise NotImplementedError


class DropoutGenerator(BaseGenerator):
    def __call__(self, words, positions):
        return [words[i] for i in range(len(words)) if i not in positions]


class BlankGenerator(BaseGenerator):
    def __init__(self, mask_token='[MASK]'):
        self.mask_token = mask_token

    def __call__(self, words, positions):
        for i in positions:
            words[i] = self.mask_token
        return words


class UnigramGenerator(BaseGenerator):
    def __init__(self, path):
        self.unigrams = []
        freqs = []
        with open(path, 'r') as f:
            for line in f:
                unigram, freq = line.rstrip().split('\t')
                self.unigrams.append(unigram)
                freqs.append(int(freq))
        freqs = np.array(freqs)
        self.weights = freqs / freqs.sum() 

    def __call__(self, words, positions):
        sub_idx = self._get_subs(len(positions))
        for p, i in zip(positions, sub_idx):
            words[p] = self.unigrams[i]
        return words

    def _get_subs(self, n):
        cumsum = np.cumsum(self.weights)
        rdm_unif = np.random.rand(n)
        return np.searchsorted(cumsum, rdm_unif)


class BigramKNGenerator(object):
    def __init__(self, path):
        table = Counter()
        with open(path, 'r') as f:
            for line in f:
                bigram  = line.rstrip().split('\t')[0]
                bigram = bigram.split(' ')
                table[bigram[1]] += 1
        self.unigrams = list(table.keys())
        freqs = np.array(list(table.values()))
        self.weights = freqs / freqs.sum()

    def __call__(self, words, positions):
        sub_idx = self._get_subs(len(positions))
        for p, i in zip(positions, sub_idx):
            words[p] = self.unigrams[i]
        return words

    def _get_subs(self, n):
        cumsum = np.cumsum(self.weights)
        rdm_unif = np.random.rand(n)
        return np.searchsorted(cumsum, rdm_unif)


class WordNetGenerator(BaseGenerator):
    from nltk.corpus import wordnet as wn
    def __init__(self, lang='jpn'):
        self.lang = lang

    def __call__(self, words, positions):
        for i in positions:
            synonym = self._get_synonym(words[i])
            if synonym is not None:
                words[i] = synonym
        return words

    def _get_synonym(self, word):
        synsets = self.wn.synsets(word, lang=self.lang)
        if len(synsets) == 0:
            return None

        synonyms = reduce(concat, [s.lemma_names(self.lang) for s in synsets])
        if len(synonyms) == 0:
            return None
        
        if self.lang == 'jpn': # cleaning verbal nouns
            synonyms = list(map(lambda x: x.replace('+', ''), synonyms))
        return random.choice(synonyms)


class Word2vecGenerator(BaseGenerator):
    def __init__(self, path, th=0.5):
        from gensim.models import KeyedVectors
        _, ext = os.path.splitext(path)
        is_binary = True if ext == '.bin' else False
        self.w2v = KeyedVectors.load_word2vec_format(path, binary=is_binary)
        self.th = th

    def __call__(self, words, positions):
        for i in positions:
            synonym = self._get_synonym(words[i])
            if synonym is not None:
                words[i] = synonym
        return words

    def _get_synonym(self, word):
        try:
            synonyms = [w for w, i in self.w2v.similar_by_word(word) if i > self.th]
        except KeyError:
            return None


        if len(synonyms) == 0:
            return None

        return random.choice(synonyms)


class PPDBGenerator(object):
    def __init__(self, path):
        self.table = {}
        with open(path, 'r') as f:
            for line in f:
                line = line.rstrip().split('\t')
                self.table[line[0]] = self.table.get(line[1], []) + [line[1]]

    def __call__(self, words, positions):
        for i in positions:
            synonym = self._get_synonym(words[i])
            if synonym is not None:
                words[i] = synonym
        return words

    def _get_synonym(self, word):
        synonyms = self.table.get(word, None)
        if synonyms is None:
            return None
        synonym = random.choice(synonyms)
        return synonym



class BertGenerator(object):
    def __init__(self, tokenizer, bert, temparature=1.0):
        self.tokenizer = tokenizer
        self.mask_token = tokenizer.vocab.mask_token
        self.cls_token = tokenizer.vocab.cls_token
        self.sep_token = tokenizer.vocab.sep_token
        self.mask_idx = tokenizer.vocab.mask_id
        self.bert = bert
        self.bert.eval()
        self.temparature=temparature

        # mask       
        special_token_ids = [tokenizer.convert_tokens_to_ids(w) for w in
            tokenizer.vocab.special_tokens_map.values()]
        self.mask = torch.ones(len(tokenizer.vocab))
        for i in special_token_ids:
            self.mask[i] = self.mask[i] * 0

    def __call__(self, words, positions):
        augmented_words = copy.deepcopy(words)
        for i in positions:
            synonym = self._get_synonym(words, i)
            augmented_words[i] = synonym
        return augmented_words

    def _get_synonym(self, words, i):
        input_tokens = [self.cls_token] + words + [self.sep_token]
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(input_tokens)
        indexed_tokens[i+1] = self.mask_idx
        tokens_tensor = torch.tensor([indexed_tokens])

        with torch.no_grad():
            outputs = self.bert(tokens_tensor)
            preds = torch.div(outputs[0][0][i+1], self.temparature)
            preds = torch.softmax(preds, dim=0)
            preds = torch.mul(preds, self.mask)
            synonym_id = random.choices(range(len(self.tokenizer.vocab)), 
                                     weights=preds.tolist())
            return self.tokenizer.convert_ids_to_tokens(synonym_id)[0]
