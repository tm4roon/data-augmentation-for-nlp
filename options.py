# -*- coding: utf-8 -*-


def sub_opts(parser):
    group = parser.add_argument_group('Sub-options')

    # augmentation
    ## smooth (unigram and bigramKN)
    group.add_argument('--unigram-frequency-for-generation', 
        '-unigram-frequency-for-generation',
        default='./data/others/sample_unigram.tsv',
        help='path to file  which contains unigram and its frequency')
    group.add_argument('--bigram-frequency-for-generation', 
        '-bigram-frequency-for-generation',
        default='./data/others/sample_bigram.tsv',
        help='path to file  which contains bigram and its frequency')
    ## wordnet
    group.add_argument('--lang-for-wordnet', '-lang-for-wordnet', default='jpn',
        help='language in wordnet')
    ## word2vec
    group.add_argument('--w2v-file', '-w2v-file',
        default='./data/others/sample_skipgram.bin',
        help='path to word embedding file, which is word2vec format')
    ## ppdb
    group.add_argument('--ppdb-file', '-ppdb-file',
        default='./data/others/sample_jppdb.tsv',
        help='path to paraphrase database, which is tsv ormat')
    ## bert
    group.add_argument('--model-name-or-path', '-model-name-or-path', 
        default='bert-base-multilingual-uncased',
        help='Path to pre-trained model or shortcut name selected')
    group.add_argument('--temparature', '-temparature', 
        type=float, default=1.0,
        help='temparature in softmax function')
    return group


def generate_opts(parser):
    group = parser.add_argument_group('Generation for Synthetic data')
    group.add_argument('--input', '-input', default='./data/samples/sample.txt',
        help='filename of the train data')
    group.add_argument('--vocab-file', '-vocab-file', 
        default='./data/vocab/bert-base-multilingual-uncased-vocab.txt',
        help='vocabulary file')
    group.add_argument('--augmentation-strategy', '-augmentation-strategy',
        choices=['dropout', 'blank', 'unigram', 'bigramkn', 'wordnet', \
                 'ppdb', 'word2vec', 'bert'], 
        default='dropout',
        help='augmentation method')
    group.add_argument('--sampling-strategy', '-sampling-strategy',
        choices=['random'], default='random',
        help='method of sampling token\'s position for augmentation')
    group.add_argument('--augmentation-rate', '-augmentation-rate', 
        type=float, default=0.2, 
        help='probability of replacing token in a sentence')
    return group
