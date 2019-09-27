# -*- coding: utf-8 -*-

import argparse
from collections import Counter
from tqdm import tqdm


def count_lines(path):
    with open(path, 'r') as f:
        return sum([1 for _ in f])

def to_ngram(words, n):
    return list(zip(*(words[i:] for i in range(n))))


def main(args):
    counter = Counter()
    n_lines = count_lines(args.input)

    with open(args.input) as fin:
        for line in tqdm(fin, total=n_lines):
            words = line.rstrip().split(' ')
            ngrams = to_ngram(words, args.ngram)
            for ngram in ngrams:
                key = ' '.join(ngram)
                counter[key] += 1
    
    for key, value in counter.most_common():
        print(key + '\t' + str(value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help='path to input file')
    parser.add_argument('--ngram', '-n', type=int, help='n-gram')
    args = parser.parse_args()
    main(args)
