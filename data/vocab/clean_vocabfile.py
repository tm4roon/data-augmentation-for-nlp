# -*- coding: utf-8 -*-

import argparse
import re


word_pattern = re.compile(r'^▁')
spm_special_tokens = ['<unk>', '<s>', '</s>']
add_special_tokens = ['[PAD]', '[UNK]', '[BOS]', '[EOS]', '[CLS]', '[SEP]', '[MASK]']

def main(args):
    # display new special tokens
    for token in add_special_tokens:
        print(token)

    # cleaning
    with open(args.input, 'r') as f:
        for line in f:
            token, _ = line.rstrip().split('\t')
            if token in spm_special_tokens:
                continue
            elif word_pattern.match(token) is None:
                print(f'##{token}')
            else:
                print(token.lstrip('▁'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--input', help='vocabulary file')
    args = parser.parse_args()
    main(args)
    
