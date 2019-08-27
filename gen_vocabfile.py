# -*- coding: utf-8 -*-

import argparse
from tqdm import tqdm
from collections import Counter

def count_lines(path):
    with open(path, 'r') as f:
        return sum([1 for _ in f])


def main(args):
    counter = Counter()
    n_lines = count_lines(args.input)

    with open(args.input, 'r') as f:
        for line in tqdm(f, total=n_lines):
            lines = line.split('\t')
            if args.mode == 'src':
                line = lines[0].rstrip()
            elif args.mode == 'tgt':
                line = lines[1].rstrip()
            elif args.mode == 'share':
                line = ' '.join(lines).rstrip()
            
            words = line.split(' ')
            for w in words:
                counter[w] += 1

    with open(args.output, 'w') as f:
        special_tokens = ['[PAD]', '[UNK]', '[BOS]', '[EOS]', '[CLS]', '[SEP]', '[MASK]']
        f.write('\n'.join(special_tokens) + '\n')

        for word, freq in counter.most_common(args.vocabsize):
            if word in special_tokens:
                continue
            if word == '':
                continue
            if freq >= args.min_freq:
                f.write(word + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='./data/samples/subwords/sample_train.bpe.tsv',
        help='input file')
    parser.add_argument('--output', default='./vocab.txt',
        help='output file name')
    parser.add_argument('--mode', choices=['src', 'tgt', 'share'], default='share',
        help='Language for generating vocabulary file')
    parser.add_argument('--min-freq', type=int, default=0,
        help='minimum frequency in a vocabulary file')
    parser.add_argument('--vocabsize', type=int, default=32000,
        help='maximum vocabulary size')
    args = parser.parse_args()
    main(args) 
