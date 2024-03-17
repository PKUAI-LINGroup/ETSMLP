# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
from typing import Dict, List, Optional,Tuple
import torch.nn.functional as F
import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq.modules.fairseq_dropout import FairseqDropout
from torch import Tensor
from .smlp_module import SMLP_module
from .sequence_norm import SequenceNorm
from fairseq.modules.quant_noise import quant_noise


class SMLPEncoderLayer(nn.Module):
    """ETSMLP Encoder layer block.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)
        self.smlp = self.build_smlp(self.embed_dim,self.embed_dim, args)
        self.norm_type=args.norm_type,
        self.norm = SequenceNorm(args.norm_type, self.embed_dim)

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0)
        # self.activation_dropout_module = FairseqDropout(
        #     float(activation_dropout_p), module_name=self.__class__.__name__
        # )
        self.normalize_before = args.encoder_normalize_before

        self.has_ffn = args.has_ffn
        if args.has_ffn:
            self.ffn_norm = SequenceNorm(args.norm_type, self.embed_dim)
            self.fc1 = self.build_fc1(
                self.embed_dim,
                args.encoder_ffn_embed_dim,
                self.quant_noise,
                self.quant_noise_block_size,
            )
            self.fc2 = self.build_fc2(
                args.encoder_ffn_embed_dim,
                self.embed_dim,
                self.quant_noise,
                self.quant_noise_block_size,
            )

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_smlp(
        self, input_dim,output_dim, args,
    ):
        return SMLP_module(input_dim,output_dim,
                    args.encoder_attention_heads,
                    q_dim=args.encoder_q_dim,
                    k_dim=args.encoder_k_dim,
                    args = args)

    def residual_connection(self, x, residual):
        return residual + x

    def forward(self, x, encoder_padding_mask, attn_mask: Optional[Tensor] = None,extra_position=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        # if attn_mask is not None:
        #     attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.norm(x)

        x = self.smlp(query=x,key_padding_mask=encoder_padding_mask)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.norm(x)
        # smlp2
        # residual = x
        # x = self.activation_fn(self.smlp2(query=x,key_padding_mask=encoder_padding_mask))
        # x = self.dropout_module(x)
        # x = self.residual_connection(x, residual)

        # ffn
        if self.has_ffn:
            residual = x
            if self.normalize_before:
                x = self.ffn_norm(x)

            x = self.activation_fn(self.fc1(x))
            x = self.dropout_module(x)
            x = self.fc2(x)
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.ffn_norm(x)

        return x


def reverse_cumsum(x, dim):
    return torch.flip(torch.cumsum(torch.flip(x, [dim]), dim), [dim])

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m
