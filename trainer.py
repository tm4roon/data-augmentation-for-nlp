# -*- coding: utf-8 -*-

import torch.nn as nn
import parallel


class Trainer(object):
    def __init__(self, model, criterion, optimizer, scheduler, clip, iteration=0):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.clip = clip
        self.n_updates = iteration

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def step(self, srcs, tgts):
        self.optimizer.zero_grad()
        slen, bsz = srcs.size()
        dec_outs = self.model(srcs, tgts[:-1])

        # translation loss
        loss = self.criterion(
            dec_outs.view(-1, dec_outs.size(2)), 
            tgts[1:].view(-1)
        )

        if self.model.training:
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()
            self.n_updates += 1
        return loss
