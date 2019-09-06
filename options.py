# -*- coding: utf-8 -*-

def train_opts(parser):
    group = parser.add_argument_group('Training')
    group.add_argument('--train', default='./data/samples/unidic/sample_train.tsv',
        help='filename of the train data')
    group.add_argument('--valid', default='./data/samples/unidic/sample_valid.tsv',
        help='filename of the validation data')
    group.add_argument('--vocab-file', default='./data/vocab/bccwj-bpe.vocab',
        help='vocabulary file')
    group.add_argument('--src-minlen', type=int, default=0,
        help='minimum sentence length of source side for training')
    group.add_argument('--tgt-minlen', type=int, default=0,
        help='minimum sentence length of target side for training')
    group.add_argument('--src-maxlen', type=int, default=1024,
        help='maximum sentence length of source side for training')
    group.add_argument('--tgt-maxlen', type=int, default=1024,
        help='maximum sentence length of target side for training')
    group.add_argument('--batch-size', type=int, default=32, 
        help='batch size')
    group.add_argument('--savedir', default='./checkpoints', 
        help='path to save models')
    group.add_argument('--max-epoch', type=int, default=30, 
        help='number of epochs')
    group.add_argument('--max-update', type=int, default=0,
        help='number of updates')
    group.add_argument('--lr', type=float, default=0.25,
        help='learning rate')
    group.add_argument('--min-lr', type=float, default=1e-5, 
        help='minimum learning rate')
    group.add_argument('--clip', type=float, default=1.0,
        help='gradient cliping')
    group.add_argument('--gpu', action='store_true',
         help='whether gpu is used')
    group.add_argument('--arch', choices=['transformer', 'translm'], 
        default='transformer',
        help='architecutre for machine translation')
    # group.add_argument('--gpus', '-gpus', type=int, nargs='*', default=[],
    #     help='list of gpu ids.')
    group.add_argument('--optimizer', choices=['sgd', 'adam', 'adamw', 'adagrad'],
        default='sgd', help='optimizer')
    group.add_argument('--save-epoch', type=int, default=10)
    return group

    
def model_opts(parser):
    group = parser.add_argument_group('Model\'s hyper-parameters')
    group.add_argument('--encoder-embed-dim', type=int, default=512,
        help='dimension of word embeddings of encoder')
    group.add_argument('--decoder-embed-dim', type=int, default=512,
        help='dimension of word embeddings of decoder')
    group.add_argument('--src-embed-path', default=None,
        help='pre-trained word embeddings of source side')
    group.add_argument('--tgt-embed-path', default=None,
        help='pre-trained word embeddings of target side')
    group.add_argument('--src-min-freq', type=int, default=0,
        help='''map words of source side appearing less than 
                threshold times to unknown''')
    group.add_argument('--tgt-min-freq', type=int, default=0,
        help='''map words of target side appearing less than
              threshold times to unknown''')
    group.add_argument('--encoder-hidden-dim', type=int, default=2048,
        help='number of hidden units per encoder layer')
    group.add_argument('--decoder-hidden-dim', type=int, default=2048,
        help='number of hidden units per decoder layer')
    group.add_argument('--encoder-layers', type=int, default=6,
        help='number of encoder layers')
    group.add_argument('--decoder-layers', type=int, default=6,
        help='number of decoder layers')
    group.add_argument('--encoder-heads', type=int, default=16,
        help='number of attention heads of encoder')
    group.add_argument('--decoder-heads', type=int, default=16,
        help='number of attention heads of decoder')
    group.add_argument('--dropout', type=float, default=0.2,
        help='dropout applied to layers (0 means no dropout)')
    group.add_argument('--activation-dropout', type=float, default=0.1,
        help='dropout after activation fucntion in self attention')
    group.add_argument('--attention-dropout', type=float, default=0.1,
        help='dropout in self attention')
    group.add_argument('--encoder-normalize-before', action='store_true',
        help='apply layernorm before each encoder block')
    group.add_argument('--decoder-normalize-before', action='store_true',
        help='apply layernorm before each decoder block')
    return group


def translate_opts(parser):
    group = parser.add_argument_group('Translation')
    group.add_argument('--model', default='./checkpoints/checkpoint_best.pt',
        help='model file for translation')
    group.add_argument('--input', default='./data/samples/sample_test.txt',
        help='input file')
    group.add_argument('--batch-size', type=int, default=32,
        help='batch size')
    group.add_argument('--maxlen', type=int, default=100,
        help='maximum length of output sentence')
    # group.add_argument('--gpu', action='store_true',
    #     help='whether gpu is used')
    group.add_argument('--gpus', '-gpus', type=int, nargs='*', default=[],
        help='list of gpu ids.')
    return group
