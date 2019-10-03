# -*- coding: utf-8 -*-

import argparse


def main(args):
    with open(args.input) as f:
        for line in f:
            splitted_line = line.rstrip().split('|||')
            src = splitted_line[0].rstrip().lstrip().split(' ')
            tgt = splitted_line[1].rstrip().lstrip().split(' ')
            if len(src) > 1:
                continue
            print(' '.join(src).replace('\\', '') + '\t' + ' '.join(tgt).replace('\\', ''))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help='path to input file')
    args = parser.parse_args()
    main(args)

