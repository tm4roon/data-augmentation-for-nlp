# -*- coding: utf-8 -*-

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional_embedding import (
    LearnedPositionalEmbedding,
    SinusoidalPositionalEmbedding
)

from .multihead_attention import MultiheadAttn
from .utils import Linear


class TransformerEncoder(nn.Module):
    def __init__(self, args, vocabsize, pad_idx):
        super(TransformerEncoder, self).__init__()
        self.vocabsize = vocabsize
        self.pad_idx = pad_idx
        self.dropout = args.dropout

        self.embed_dim = args.encoder_embed_dim
        self.n_layers = args.encoder_layers

        self.w_embed = nn.Embedding(self.vocabsize, self.embed_dim)
        #    if self.field.vocab.vectors is None \
        #    else nn.Embedding.from_pretrained(self.field.vocab.vectors, freeze=True)
        self.p_embed = SinusoidalPositionalEmbedding(self.embed_dim, self.pad_idx)
        self.embed_scale = math.sqrt(self.embed_dim)
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(args) for _ in range(self.n_layers)])

    def forward(self, inputs):
        # embed tokens and positions
        x = self.embed_scale * self.w_embed(inputs)
        x += self.p_embed(inputs)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # padding mask
        encoder_pad_mask = inputs.eq(self.pad_idx)
        if not encoder_pad_mask.any():
            encoder_pad_mask = None

        # encoder layers 
        for layer in self.layers:
            x = layer(x, encoder_pad_mask)
        return {'enc_out': x, 'enc_pad_mask': encoder_pad_mask}


class TransformerEncoderLayer(nn.Module):
    def __init__(self, args):
        super(TransformerEncoderLayer, self).__init__()
        self.embed_dim = args.encoder_embed_dim
        self.hidden_dim = args.encoder_hidden_dim
        self.n_heads = args.encoder_heads

        self.dropout = args.dropout
        self.attention_dropout = args.attention_dropout
        self.activation_dropout = args.activation_dropout

        self.self_attn = MultiheadAttn(
            self.embed_dim, self.n_heads, dout=self.attention_dropout)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.normalize_before = args.encoder_normalize_before
        self.fc1 = Linear(self.embed_dim, self.hidden_dim)
        self.fc2 = Linear(self.hidden_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x, encoder_pad_mask):
        residual = x
        x = self.maybe_normalize(self.self_attn_layer_norm, x, before=True)
        x, _ = self.self_attn(
            query=x, key=x, value=x, key_pad_mask=encoder_pad_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_normalize(self.self_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_normalize(self.final_layer_norm, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)
        x = self.maybe_normalize(self.final_layer_norm, x, after=True)
        return x

    def maybe_normalize(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x
    
