# -*- coding: utf-8 -*-

import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import TransformerEncoder
from .decoder import TransformerDecoder


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, bos_idx):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.bos_idx = bos_idx

    def forward(self, srcs, prev_tokens):
        enc_outs = self.encoder(srcs)
        return self.decoder(prev_tokens, enc_outs)

    def generate(self, srcs, maxlen):
        slen, bsz = srcs.size()
        enc_outs = self.encoder(srcs)
        
        prev_tokens = torch.ones_like(srcs[0]).unsqueeze(0) * self.bos_idx
        while len(prev_tokens) < maxlen+1:
            output_tokens = self.decoder(
                prev_tokens, enc_outs, incremental_state=True)
            output_tokens = output_tokens.max(2)[1][-1].unsqueeze(0)
            prev_tokens = torch.cat((prev_tokens, output_tokens), 0)
        return prev_tokens
