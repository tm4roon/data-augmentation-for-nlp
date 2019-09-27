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
    

    # scheduling
    ## step
    group.add_argument('--step-size', '-step-size', type=int, default=10,
        help='step size when augmentation rate is decayed')
    group.add_argument('--decay', '-decay', type=float, default=0.5,
        help='decay of augmentation rate per step-size')
    ## warmup-constant, warmup-linear
    group.add_argument('--warmup-epoch', '-warmup-epoch', type=int, default=3,
        help='Number of epoch for warm-up.')
    return group


def train_opts(parser):
    group = parser.add_argument_group('Training options')

    # data 
    group.add_argument('--savedir', '-savedir', default='./checkpoints', 
        help='path to save models')
    group.add_argument('--train', '-train', default='./data/samples/sample_train.tsv',
        help='filename of the train data')
    group.add_argument('--valid', '-valid', default='./data/samples/sample_valid.tsv',
        help='filename of the validation data')
    group.add_argument('--vocab-file', '-vocab-file', default='./data/vocab/bccwj-bpe.vocab',
        help='vocabulary file')
    group.add_argument('--src-minlen', '-src-minlen', type=int, default=0,
        help='minimum sentence length of source side for training')
    group.add_argument('--tgt-minlen', '-tgt-minlen', type=int, default=0,
        help='minimum sentence length of target side for training')
    group.add_argument('--src-maxlen', '-src-maxlen', type=int, default=1024,
        help='maximum sentence length of source side for training')
    group.add_argument('--tgt-maxlen', '-tgt-maxlen', type=int, default=1024,
        help='maximum sentence length of target side for training')

    # model 
    group.add_argument('--arch', '-arch', choices=['transformer', 'translm'], 
        default='transformer',
        help='architecutre for machine translation')

    # training 
    group.add_argument('--batchsize', '-batchsize', type=int, default=32, 
        help='batch size')
    group.add_argument('--max-epoch', '-max-epoch', type=int, default=30, 
        help='number of epochs')
    group.add_argument('--save-epoch', '-save-epoch', type=int, default=10)
    group.add_argument('--lr', '-lr', type=float, default=1e-4,
        help='learning rate')
    group.add_argument('--clip', '-clip', type=float, default=1.0,
        help='gradient cliping')
    group.add_argument('--optimizer', '-optimizer', default='adam',
        choices=['sgd', 'adam', 'adamw', 'adagrad'],
        help='optimizer')
    group.add_argument('--lr-scheduler', '-lr-scheduler', default='constant',
        choices=['constant', 'warmup_constant', 'warmup_linear'], 
        help='learning rate scheduler')
    group.add_argument('--gpu', '-gpu', action='store_true',
         help='whether gpu is used')

    # data-augmentation
    group.add_argument('--augmentation-strategy', '-augmentation-strategy',
        choices=['base', 'dropout', 'blank', 'unigram', 'bigramkn', 'wordnet', \
                 'ppdb', 'word2vec', 'bert'], 
        default='base',
        help='augmentation method')
    group.add_argument('--sampling-strategy', '-sampling-strategy',
        choices=['random', 'absolute_discounting'], default='random',
        help='method of sampling token\'s position for augmentation')
    group.add_argument('--ar-scheduler', '-ar-scheduler',
        default='constant', 
        choices=['constant', 'linear', 'exp', 'step', 'warmup_constant', 'warmup_linear'],
        help='scheduler of replacing probability')
    group.add_argument('--side', '-side', default='src',
        choices=['src', 'tgt', 'both'],
        help='side augmented data, source language or target language')
    group.add_argument('--augmentation-rate', '-augmentation-rate', 
        type=float, default=0.1, 
        help='probability of replacing token in a sentence')
    return group

    
def model_opts(parser):
    group = parser.add_argument_group('Model\'s hyper-parameters')
    group.add_argument('--encoder-embed-dim', '-encoder-embed-dim', 
        type=int, default=512,
        help='dimension of word embeddings of encoder')
    group.add_argument('--decoder-embed-dim', '-decoder-embed-dim', 
        type=int, default=512,
        help='dimension of word embeddings of decoder')
    group.add_argument('--encoder-hidden-dim', '-encoder-hidden-dim',
        type=int, default=2048,
        help='number of hidden units per encoder layer')
    group.add_argument('--decoder-hidden-dim', '-decoder-hidden-dim',
        type=int, default=2048,
        help='number of hidden units per decoder layer')
    group.add_argument('--encoder-layers', '-encoder-layers', 
        type=int, default=6,
        help='number of encoder layers')
    group.add_argument('--decoder-layers', '-decoder-layers', 
        type=int, default=6,
        help='number of decoder layers')
    group.add_argument('--encoder-heads', '-encoder-heads', 
        type=int, default=16,
        help='number of attention heads of encoder')
    group.add_argument('--decoder-heads', '-decoder-heads',
        type=int, default=16,
        help='number of attention heads of decoder')
    group.add_argument('--dropout', '-dropout', 
        type=float, default=0.2,
        help='dropout applied to layers (0 means no dropout)')
    group.add_argument('--activation-dropout', '-activation-dropout',
        type=float, default=0.1,
        help='dropout after activation fucntion in self attention')
    group.add_argument('--attention-dropout', '-attention-dropout', 
        type=float, default=0.1,
        help='dropout in self attention')
    group.add_argument('--encoder-normalize-before', '-encoder-normalize-before', 
        action='store_true',
        help='apply layernorm before each encoder block')
    group.add_argument('--decoder-normalize-before', '-decoder-normalize-before',
        action='store_true',
        help='apply layernorm before each decoder block')
    return group


def translate_opts(parser):
    group = parser.add_argument_group('Translation options')
    group.add_argument('--model', '-model',
        default='./checkpoints/checkpoint_best.pt',
        help='model file for translation')
    group.add_argument('--input', '-input', 
        default='./data/samples/sample_test.txt',
        help='input file')
    group.add_argument('--batchsize', '-batchsize', 
        type=int, default=32,
        help='batch size')
    group.add_argument('--maxlen', '-maxlen', 
        type=int, default=100,
        help='maximum length of output sentence')
    group.add_argument('--gpu', '-gpu', action='store_true',
         help='whether gpu is used')
    return group
