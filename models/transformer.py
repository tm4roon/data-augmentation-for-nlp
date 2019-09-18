# -*- coding: utf-8 -*-

import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import TransformerEncoder
from .decoder import TransformerDecoder
from .utils import fill_ninf


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


class TranslationLM(TransformerDecoder):
    def __init__(self, args, vocabsize, pad_idx, bos_idx, sep_idx):
        super().__init__(args, vocabsize, pad_idx, no_enc_attn=True)
        self.bos_idx = bos_idx
        self.sep_idx = sep_idx

    def forward(self, srcs, tgts=None, incremental_state=None):
        inputs = srcs
        positions = self.p_embed(inputs, incremental_state=None)

        # for translation
        if tgts is not None:
            ones = torch.ones_like(srcs[0].unsqueeze(0))
            tpos = self.p_embed(
                torch.cat((ones*self.pad_idx, tgts)),
                incremental_state=incremental_state
            )
            positions = torch.cat((positions, tpos))
            inputs = torch.cat((inputs, ones*self.sep_idx, tgts))

        x = self.w_embed(inputs)
        x *= self.embed_scale
        x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # padding mask
        decoder_pad_mask = inputs.eq(self.pad_idx)
        if not decoder_pad_mask.any():
            decoder_pad_mask = None

        delim_idx = 1 if tgts is None else len(srcs) + 1
        self_attn_mask = self.buffered_future_mask(x, delim_idx)

        # decoder layers
        for layer in self.layers:
            x = layer(
                x, 
                None,
                None,
                self_attn_mask,
                decoder_pad_mask,
                incremental_state,
           )
        x = self.out_projection(x)
        delim = 0 if tgts is None else srcs.size(0) + 1
        return x[delim:]

    def buffered_future_mask(self, tensor, delim_idx=1):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None:
            self._future_mask = torch.triu(fill_ninf(tensor.new(dim, dim)), delim_idx)
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(
                fill_ninf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def generate(self, srcs, maxlen):
        slen, bsz = srcs.size()
        
        prev_tokens = torch.cat(
            (srcs,
             torch.ones_like(srcs[0]).unsqueeze(0) * self.sep_idx,
             torch.ones_like(srcs[0]).unsqueeze(0) * self.bos_idx
            ))

        while len(prev_tokens) - slen < maxlen:
            output_tokens = self.forward(
                prev_tokens, incremental_state=True)
            output_tokens = output_tokens.max(2)[1][-1].unsqueeze(0)
            prev_tokens = torch.cat(
                (prev_tokens, output_tokens), 0)
        return prev_tokens[slen+1:]
