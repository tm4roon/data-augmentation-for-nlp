# -*- coding: utf-8 -*-


def sub_opts(parser):
    group = parser.add_argument_group('Sub-options')

    # augmentation
    ## smooth (unigram and bigramKN)
    group.add_argument('--unigram-frequency-for-generation', 
        '-unigram-frequency-for-generation',
        default='./data/others/bccwj-unidic-unigram.tsv',
        help='path to file  which contains unigram and its frequency')
    group.add_argument('--bigram-frequency-for-generation', 
        '-bigram-frequency-for-generation',
        default='./data/others/bccwj-unidic-bigram.tsv',
        help='path to file  which contains bigram and its frequency')
    ## word2vec
    group.add_argument('--w2v-file', '-w2v-file',
        default='./data/others/bccwj-skipgram.bin',
        help='path to word embedding file, which is word2vec format')
    ## ppdb
    group.add_argument('--ppdb-file', '-ppdb-file',
        default='./data/others/jppdb-m.tsv',
        help='path to paraphrase database, which is tsv ormat')
    ## bert
    group.add_argument('--model', '-model', default='./path/to/bert/',
        help='path to BERT model file')

    # sampling
    ## uif
    group.add_argument('--unigram-frequency-for-uif', '-unigram-frequency-for-uif',
        default='./data/others/jawiki-unigram-frequency.tsv',
        help='path to file  which contains unigram and its frequency')
    group.add_argument('--bigram-frequency-for-sampling', 
        '-bigram-frequency-for-sampling',
        default='./data/others/bccwj-unidic-bigram.tsv',
        help='path to file  which contains bigram and its frequency')
    return group


def generate_opts(parser):
    group = parser.add_argument_group('Generation for Synthetic data')
    group.add_argument('--input', '-input', default='./data/samples/sample_train.txt',
        help='filename of the train data')
    group.add_argument('--vocab-file', '-vocab-file', 
        default='./data/vocab/bccwj-bpe.vocab',
        help='vocabulary file')
    group.add_argument('--augmentation-strategy', '-augmentation-strategy',
        choices=['dropout', 'blank', 'unigram', 'bigramkn', 'wordnet', \
                 'ppdb', 'word2vec', 'bert'], 
        default='dropout',
        help='augmentation method')
    group.add_argument('--sampling-strategy', '-sampling-strategy',
        choices=['random', 'absolute_discounting'], default='random',
        help='method of sampling token\'s position for augmentation')
    group.add_argument('--augmentation-rate', '-augmentation-rate', 
        type=float, default=0.2, 
        help='probability of replacing token in a sentence')
    return group
