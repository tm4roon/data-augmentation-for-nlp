# -*- coding: utf-8 -*-

"""
Dynamic data augmentation for sentence rewriting tasks such as summarization,
grammatical error correction, text simplification, paraphrase generation and
style transfer. Using `--arch` option, you can choose either transformer
(Vaswani et al., 2017) or translation language model (Khandelwal et al. 2019,
Hoang et al. 2019) for sentence rewriting. Also, you are able to augmentation
way from the following six methods: dropout, blank, smooth, wordnet, word2vec, 
and bert.
"""

import argparse
import math
import json
import os
from collections import OrderedDict

from tqdm import tqdm
from tqdm import trange

import torch
import torch.nn as nn
import torch.optim as optim

import pytorch_transformers
from pytorch_transformers import BertTokenizer as Tokenizer

import options

from augmentation_module.dataset import (
    Dataset,
    DataAugmentationIterator,
)
from augmentation_module import augmentor
from augmentation_module import generator
from augmentation_module import sampler
from augmentation_module import scheduler

from models.encoder import TransformerEncoder
from models.decoder import TransformerDecoder
from models.transformer import (
    Transformer,
    TranslationLM,
)


def display_stats(stats):
    for k in stats.keys():
        coverage = 100 * (stats[k]['n_tokens'] - stats[k]['n_unks']) / stats[k]['n_tokens']
        print(f"| [{k}] {stats[k]['n_tokens']} tokens,", end='')
        print(f" coverage: {coverage:.{4}}%")
    print('')


def step(tokenizer, model, iterator, criterion, optimizer, lr_scheduler, device):
        pbar = tqdm(iterator, dynamic_ncols=True) if model.training else iterator
        total_loss = 0.0
        for srcs, tgts in pbar:
            optimizer.zero_grad()
            srcs = srcs.to(device)
            tgts = tgts.to(device)
            dec_outs = model(srcs, tgts[:-1])
            loss = criterion(
                dec_outs.view(-1, dec_outs.size(2)), 
                tgts[1:].view(-1)
            )
            total_loss += loss.item()

            if model.training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                lr_scheduler.step()

                # setting of progressbar
                augmentation_rate = iterator.augmentation_rate \
                    if iterator.augmentor is not None else 0.0

                pbar.set_description(f'epoch {str(iterator.current_epoch).zfill(3)}')
                progress_state = OrderedDict(
                    loss=loss.item(),
                    ppl=math.exp(loss.item()),
                    bsz=srcs.size(1),
                    lr=optimizer.param_groups[0]['lr'],
                    ar=augmentation_rate,
                    nups=iterator.n_update,
                )
                pbar.set_postfix(progress_state)
        
        if model.training:
            pbar.close()

        total_loss /= len(iterator)

        mode = 'train' if model.training else 'valid'
        print(f'| {mode} ', end='') 
        print(f'| loss {total_loss:.{4}} ', end='')
        print(f'| ppl {math.exp(total_loss):.{4}} |')

        return total_loss


def main(args):
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available else 'cpu')

    # set tokenizer
    tokenizer = Tokenizer(
        args.vocab_file,
        do_basic_tokenize=False, 
        bos_token='[BOS]',
        eos_token='[EOS]',
        unk_token='[UNK]',
        sep_token='[SEP]',
        pad_token='[PAD]',
    )

    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)

    tokenizer.save_vocabulary(args.savedir)

    # save configure
    with open(os.path.join(args.savedir, 'config.json'), 'w') as fo:
        json.dump(vars(args), fo, indent=2)

    # load dataset for training and validation
    train_data = Dataset(
        args.train, tokenizer, 
        args.src_minlen, args.src_maxlen,
        args.tgt_minlen, args.tgt_maxlen, 
    )

    valid_data = Dataset(
        args.valid, tokenizer, 
        args.src_minlen, args.src_maxlen,
        args.tgt_minlen, args.tgt_maxlen,
    )

    # display data statistics
    print(f'| [share] Vocabulary: {len(tokenizer)} types')
    print('')

    train_stats = train_data.state_statistics()
    valid_stats = valid_data.state_statistics()
    for name, stats in [('train', train_stats), ('valid', valid_stats)]:
        file_path = args.train if name == 'train' else args.valid
        print(f'{name}: {file_path}')
        display_stats(stats)

    # set data-augmentation module
    to_word = False


    if args.sampling_strategy == 'random':
        sampling_fn = sampler.UniformSampler()
    elif args.sampling_strategy == 'absolute_discounting':
        sampling_fn = sampler.AbsDiscountSampler(
        args.bigram_frequency_for_sampling)
        to_word = True


    if args.augmentation_strategy == 'base':
        augmentor_fn = None
    else:
        if args.augmentation_strategy == 'dropout':
            generator_fn = generator.DropoutGenerator()
        elif args.augmentation_strategy == 'blank':
            generator_fn = generator.BlankGenerator(
                mask_token=tokenizer.mask_token)
        elif args.augmentation_strategy == 'unigram':
            generator_fn = generator.UnigramGenerator(
                args.unigram_frequency_for_generation)
            to_word=True
        elif args.augmentation_strategy == 'bigramkn':
            generator_fn = generator.BigramKNGenerator(
                args.bigram_frequency_for_generation)
            to_word=True
        elif args.augmentation_strategy == 'wordnet':
            generator_fn = generator.WordNetGenerator(lang='jpn')
            to_word = True
        elif args.augmentation_strategy == 'word2vec':
            generator_fn = generator.Word2vecGenerator(args.w2v_file)
            to_word = True
        elif args.augmentation_strategy == 'bert':
            generator_fn = generator.BertGenerator()
        augmentor_fn = augmentor.ReplacingAugmentor(
            tokenizer, sampling_fn, generator_fn, to_word=to_word)

    # set iterator
    train_iter = DataAugmentationIterator(
        data=train_data,
        batchsize=args.batchsize,
        side=args.side,
        augmentor=augmentor_fn,
        shuffle=True,
    )

    valid_iter = DataAugmentationIterator(
        data=valid_data,
        batchsize=args.batchsize,
        augmentor=None,
        shuffle=False,
    )

    # set scheduler for dynamic data-augmentation
    if args.ar_scheduler == 'constant':
        ar_scheduler = scheduler.ConstantAR(
            iterator=train_iter, 
            augmentation_rate=args.augmentation_rate,
        )
    elif args.ar_scheduler == 'linear':
        ar_scheduler = scheduler.LinearAR(
            iterator=train_iter,
            augmentation_rate=args.augmentation_rate, 
            max_epoch=args.max_epoch
        )
    elif args.ar_scheduler == 'exp':
        ar_scheduler = scheduler.ExponentialAR(
            iterator=train_iter,
            augmentation_rate=args.augmentation_rate,
        )
    elif args.ar_scheduler == 'step':
        ar_scheduler = scheduler.StepAR(
            iterator=train_iter,
            augmentation_rate=args.augmentation_rate,
            step_size=args.step_size,
            decay=args.decay,
        )
    elif args.ar_scheduler == 'warmup_constant':
        ar_scheduler = scheduler.WarmupConstantAR(
            iterator=train_iter,
            augmentation_rate=args.augmentation_rate,
            warmup_epoch=args.warmup_epoch,
            total_epoch=args.max_epoch,
        )
    elif args.ar_scheduler == 'warmup_linear':
        ar_scheduler = scheduler.WarmupLinearAR(
            iterator=train_iter,
            augmentation_rate=args.augmentation_rate,
            warmup_epoch=args.warmup_epoch,
            total_epoch=args.max_epoch,
        )
    else:
        raise NotImplementedError

    pad_idx = tokenizer.pad_token_id
    bos_idx = tokenizer.bos_token_id

    # build model
    if args.arch == 'transformer':
        encoder = TransformerEncoder(args, len(tokenizer), pad_idx)
        decoder = TransformerDecoder(args, len(tokenizer), pad_idx)
        model = Transformer(encoder, decoder, bos_idx).to(device)
    elif args.arch == 'translm':
        sep_idx = tokenizer.sep_token_id
        model = TranslationLM(args, len(tokenizer), pad_idx, bos_idx, sep_idx).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # set optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer== 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98),
            eps=1e-09, weight_decay=0, amsgrad=False)
    else:
        raise NotImplementedError


    # set scheduler
    t_total = args.max_epoch * len(train_iter)
    warmup_steps = math.floor(t_total * 0.04)
    if args.lr_scheduler == 'constant':
        lr_scheduler = pytorch_transformers.ConstantLRSchedule(optimizer)
    elif args.lr_scheduler == 'warmup_constant':
        lr_scheduler = pytorch_transformers.WarmupConstantSchedule(
            optimizer, warmup_steps=warmup_steps)
    elif args.lr_scheduler == 'warmup_linear':
        lr_scheduler = pytorch_transformers.WarmupLinearSchedule(
            optimizer, warmup_steps=warmup_steps, t_total=t_total)
    else:
        raise NotImplementedError


    print('=============== MODEL ===============')
    print(model)
    print('')
    print('=============== OPTIMIZER ===============')
    print(optimizer)
    print('')

    epoch = 1
    max_epoch = args.max_epoch or math.inf
    best_loss = math.inf

    while epoch <= max_epoch:
        # train
        model.train()
        train_loss = step(tokenizer, model, train_iter, criterion, optimizer, 
            lr_scheduler, device)

        model.eval()
        valid_loss = step(tokenizer, model, valid_iter, criterion, optimizer, 
            lr_scheduler, device)

        # saving model
        save_vars = {
            'epoch': epoch,
            'best_loss': valid_loss if valid_loss < best_loss else best_loss,
            'args': args,
            'weights': model.state_dict()
        }

        if valid_loss < best_loss:
            filename = os.path.join(args.savedir, 'checkpoint_best.pt') 
            torch.save(save_vars, filename)
        if epoch % args.save_epoch == 0:
            filename = os.path.join(args.savedir, f'checkpoint_{epoch}.pt') 
            torch.save(save_vars, filename)
        filename = os.path.join(args.savedir, 'checkpoint_last.pt') 
        torch.save(save_vars, filename)

        epoch += 1
        ar_scheduler.step()

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Dyanamic data augmentation for sentence rewriting tasks'
    )

    options.train_opts(parser)
    options.model_opts(parser)
    options.sub_opts(parser)
    args = parser.parse_args()
    main(args)
