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

import options
import utils
from trainer import Trainer

from data_handler import (
    Dataset,
    DataAugmentationIterator,
)

from models.encoder import TransformerEncoder
from models.decoder import TransformerDecoder
from models.transformer import Transformer


def main(args):
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available else 'cpu')

    tokenizer = Tokenizer(
        args.vocab_file,
        do_basic_tokenize=False, 
        bos_token='[BOS]',
        eos_token='[EOS]'
    )
    tokenizer.save_vocabulary(args.savedir)

    # construct Field objects
    train_data = data_loader(
        tokenizer, args.train, 
        args.src_minlen, args.src_maxlen,
        args.tgt_minlen, args.tgt_maxlen,
    )

    valid_data = data_loader(
        tokenizer, args.valid,
        args.src_minlen, args.src_maxlen,
        args.tgt_minlen, args.tgt_maxlen,
    )

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

    print(f'| [share] Vocabulary: {len(tokenizer)} types')
    print('')

    train_stats = train_iter.state_statics()
    valid_stats = valid_iter.state_statics()

    for name, stats in [('train', train_stats), ('valid', valid_stats)]:
        file_path = args.train if name == 'train' else args.valid
        print(f'{name}: {file_path}')
        for k in stats.keys():
            coverage = 100 * (stats[k]['n_tokens'] - stats[k]['n_unks']) / stats[k]['n_tokens']
            print(f"| [{k}] {stats[k]['n_tokens']} tokens,", end='')
            print(f" coverage: {coverage:.{4}}%")
        print('')

    pad_idx = tokenizer.pad_token_id
    bos_idx = tokenizer.bos

    if args.arch == 'transformer':
        encoder = TransformerEncoder(args, len(tokenizer), pad_idx)
        decoder = TransformerDecoder(args, len(tokenizer), pad_idx)
        model = Transformer(encoder, decoder, bos_idx).to(device)
    elif args.arch == 'lm':
        model = TranslationLM(args, len(tokenizer), pad_idx, bos_idx, sep_idx)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    optimizer_fn = utils.get_optimizer(args.optimizer)
    optimizer = optimizer_fn(model.parameters(), lr=args.lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)

    trainer = Trainer(model, criterion, optimizer, scheduler, args.clip)

    print('=============== MODEL ===============')
    print(model)
    print('')
    print('=============== OPTIMIZER ===============')
    print(optimizer)
    print('')

    epoch = 1
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    best_loss = math.inf

    while epoch < max_epoch and trainer.n_updates < max_update and args.min_lr < trainer.get_lr():
        # train
        with tqdm(train_iter, dynamic_ncols=True) as pbar:
            train_loss = 0.0
            trainer.model.train()
            for srcs, tgts in pbar:
                bsz = srcs.size(1)
                srcs = srcs.to(device)
                tgts = tgts.to(device)
                loss = trainer.step(srcs, tgts)
                train_loss += loss.item()

                # setting of progressbar
                pbar.set_description(f"epoch {str(epoch).zfill(3)}")
                progress_state = OrderedDict(
                    loss=loss.item(),
                    ppl=math.exp(loss.item()),
                    bsz=bsz,
                    lr=trainer.get_lr(), 
                    clip=args.clip, 
                    num_updates=trainer.n_updates)
                pbar.set_postfix(progress_state)
        train_loss /= len(train_iter)

        print(f"| epoch {str(epoch).zfill(3)} | train ", end="") 
        print(f"| loss {train_loss:.{4}} ", end="")
        print(f"| ppl {math.exp(train_loss):.{4}} ", end="")
        print(f"| lr {trainer.get_lr():.1e} ", end="")
        print(f"| clip {args.clip} ", end="")
        print(f"| num_updates {trainer.n_updates} |")
        
        # validation
        valid_loss = 0.0
        trainer.model.eval()
        for srcs, tgts in valid_iter:
            bsz = srcs.size(1)
            srcs = srcs.to(device)
            tgts = tgts.to(device)
            loss = trainer.step(srcs, tgts)
            valid_loss += loss.item()
        valid_loss /= len(valid_iter)

        print(f"| epoch {str(epoch).zfill(3)} | valid ", end="") 
        print(f"| loss {valid_loss:.{4}} ", end="")
        print(f"| ppl {math.exp(valid_loss):.{4}} ", end="")
        print(f"| lr {trainer.get_lr():.1e} ", end="")
        print(f"| clip {args.clip} ", end="")
        print(f"| num_updates {trainer.n_updates} |")

        # saving model
        save_vars = {
            'epoch': epoch,
            'iteration': trainer.n_updates,
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
        trainer.scheduler.step(valid_loss)
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
