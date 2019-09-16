# -*- coding: utf-8 -*-

import argparse
import math
import os
from collections import OrderedDict

from tqdm import tqdm
from tqdm import trange

import torch
import torch.nn as nn
import torch.optim as optim

from pytorch_transformers import BertTokenizer as Tokenizer
from pytorch_transformers import AdamW

import options

from dataset import (
    Dataset,
    DataAugmentationIterator,
)

from models.encoder import TransformerEncoder
from models.decoder import TransformerDecoder
from models.transformer import (
    Transformer,
    TranslationLM,
)


def get_optimizer(method):
    if method == 'sgd':
        return optim.SGD
    elif method == 'adam':
        return optim.Adam
    elif method == 'adamw':
        return AdamW
    elif method == 'adagrad':
        return optim.Adagrad


def step(epoch, model, iterator, criterion, optimizer,  device):
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

                # setting of progressbar
                pbar.set_description(f'epoch {str(epoch).zfill(3)}')
                progress_state = OrderedDict(
                    loss=loss.item(),
                    ppl=math.exp(loss.item()),
                    bsz=srcs.size(1),
                    lr=optimizer.param_groups[0]['lr'], 
                    clip=args.clip)
                pbar.set_postfix(progress_state)
        
        if model.training:
            pbar.close()

        total_loss /= len(iterator)

        mode = 'train' if model.training else 'valid'
        print(f'| epoch {str(epoch).zfill(3)} | {mode} ', end='') 
        print(f'| loss {total_loss:.{4}} ', end='')
        print(f'| ppl {math.exp(total_loss):.{4}} ', end='')
        print('')

        return total_loss


def main(args):
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available else 'cpu')

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

    # construct Field objects
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
        for k in stats.keys():
            coverage = 100 * (stats[k]['n_tokens'] - stats[k]['n_unks']) / stats[k]['n_tokens']
            print(f"| [{k}] {stats[k]['n_tokens']} tokens,", end='')
            print(f" coverage: {coverage:.{4}}%")
        print('')

    # set iterator
    train_iter = DataAugmentationIterator(
        data=train_data,
        batchsize=args.batch_size,
        # augmentor=augmentor,
        shuffle=True,
        repeat=False,
    )

    valid_iter = DataAugmentationIterator(
        data=valid_data,
        batchsize=args.batch_size,
        augmentor=None,
        shuffle=False,
        repeat=False,
    )

    pad_idx = tokenizer.pad_token_id
    bos_idx = tokenizer.bos_token_id

    if args.arch == 'transformer':
        encoder = TransformerEncoder(args, len(tokenizer), pad_idx)
        decoder = TransformerDecoder(args, len(tokenizer), pad_idx)
        model = Transformer(encoder, decoder, bos_idx).to(device)
    elif args.arch == 'translm':
        sep_idx = tokenizer.sep_token_id
        model = TranslationLM(args, len(tokenizer), pad_idx, bos_idx, sep_idx).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    optimizer_fn = get_optimizer(args.optimizer)
    optimizer = optimizer_fn(model.parameters(), lr=args.lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)

    print('=============== MODEL ===============')
    print(model)
    print('')
    print('=============== OPTIMIZER ===============')
    print(optimizer)
    print('')

    epoch = 1
    max_epoch = args.max_epoch or math.inf
    best_loss = math.inf

    while epoch < max_epoch and args.min_lr < optimizer.param_groups[0]['lr']:
        # train
        model.train()
        train_loss = step(epoch, model, train_iter, criterion, optimizer, device)

        model.eval()
        valid_loss = step(epoch, model, valid_iter, criterion, optimizer, device)

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

        # update
        scheduler.step(valid_loss)
        epoch += 1

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser('''
        An Implimentation of Transformer.
        Attention is all you need. 
        Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
        Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. 
        In Advances in Neural Information Processing Systems, pages 6000–6010.
    ''')

    options.train_opts(parser)
    options.model_opts(parser)
    args = parser.parse_args()
    main(args)
