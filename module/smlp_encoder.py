# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn
from fairseq.modules import (
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding
)
from .real_number_embedding import RealNumberEmbedding
from .smlp_encoder_layer import SMLPEncoderLayer
from .sequence_norm import SequenceNorm


class SMLPSentenceEncoder(nn.Module):
    """
    Implementation for a ETSMLP encoder.

    Input:
        - tokens: B x T matrix representing sentences
        - segment_labels: B x T matrix representing segment label for tokens

    Output:
        - a tuple of the following:
            - a list of internal model states used to compute the
              predictions where each tensor has shape T x B x C
            - sentence representation associated with first input token
              in format B x C.
    """

    def __init__(
        self,
        args,
        padding_idx: int,
        vocab_size: int,
        num_encoder_layers: int = 6,
        embedding_dim: int = 768,
        embedding_type: str = 'sparse',
        dropout: float = 0.1,
        layerdrop: float = 0.0,
        max_seq_len: int = 256,
        use_position_embeddings: bool = True,
        offset_positions_by_padding: bool = True,
        encoder_normalize_before: bool = False,
        apply_bert_init: bool = False,
        learned_pos_embedding: bool = True,
        embed_scale: float = None,
        export: bool = False,
        traceable: bool = False,
        sen_rep_type= 'cls',
        freeze=False,
    ) -> None:
        super().__init__()
        self.args=args
        self.padding_idx = padding_idx
        self.vocab_size = vocab_size
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.layerdrop = layerdrop
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.use_position_embeddings = use_position_embeddings
        self.apply_bert_init = apply_bert_init
        self.learned_pos_embedding = learned_pos_embedding
        self.traceable = traceable
        self.tpu = False  # whether we're on TPU
        self.sen_rep_type = sen_rep_type
        assert embedding_type in ['sparse', 'linear']
        self.embedding_type=embedding_type
        self.embed_tokens = self.build_embedding(embedding_type,
            self.vocab_size, self.embedding_dim, self.padding_idx
        )
        self.embed_scale = embed_scale


        self.embed_positions = (
            PositionalEmbedding(
                self.max_seq_len,
                self.embedding_dim,
                padding_idx=(self.padding_idx if offset_positions_by_padding else None),
                learned=self.learned_pos_embedding,
            )
            if self.use_position_embeddings
            else None
        )

        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.num_layers = num_encoder_layers
        self.layers.extend(
            [
                self.build_smlp_sentence_encoder_layer(self.args)
                for _ in range(num_encoder_layers)
            ]
        )

        self.emb_layer_norm = SequenceNorm(args.norm_type, self.embedding_dim)
        # self.emb_layer_norm = None

        if encoder_normalize_before:
            self.final_norm = SequenceNorm(args.norm_type, embedding_dim, export=export)
        else:
            self.final_norm = None

        self.freeze = freeze
        def freeze_module_params(m):
            # if m is not None:
            #     for p in m.parameters():
            #         p.requires_grad = False
            m.requires_grad = False

        if self.freeze:
            for layer in range(num_encoder_layers):
                freeze_module_params(self.layers[layer].smlp.smlp_cumsum.log_delta)

    # def build_embedding(self, vocab_size, embedding_dim, padding_idx):
    #     #     return nn.Embedding(vocab_size, embedding_dim, padding_idx)

    def build_embedding(self,embedding_type, vocab_size, embedding_dim, padding_idx):
        if embedding_type == 'sparse':
            embed_tokens = Embedding(vocab_size, embedding_dim, padding_idx)
            return embed_tokens
        else:
            embed_tokens = RealNumberEmbedding(embedding_dim)
            return embed_tokens

    def build_smlp_sentence_encoder_layer(self,args):
        return SMLPEncoderLayer(args)


    def forward(
        self,
        tokens: torch.Tensor,
        src_lengths: torch.Tensor,
        last_state_only: bool = False,
        positions: Optional[torch.Tensor] = None,
    ) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], torch.Tensor]:

        if self.embedding_type == 'sparse':
            padding_mask = tokens.eq(self.padding_idx)
            if not self.traceable and not self.tpu and not padding_mask.any():
                padding_mask = None
            # B x T -> B x T x D
            x = self.embed_tokens(tokens)
        else:
            padding_mask = None
            # B x T -> B x T x D
            x = self.embed_tokens(tokens)


        if self.embed_positions is not None:
            x = x + self.embed_positions(tokens, positions=positions)

        # if self.quant_noise is not None:
        #     x = self.quant_noise(x)
        # assert self.emb_layer_norm is None
        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        x = self.dropout_module(x)

        # account for padding while computing the representation
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)


        for i in range(self.num_layers):
            x = self.layers[i](x, encoder_padding_mask=padding_mask)
            if not last_state_only:
                inner_states.append(x)

        if self.final_norm is not None:
            x = self.final_norm(x)

        if padding_mask is not None:
            x = x.masked_fill(padding_mask.transpose(0, 1).unsqueeze(-1), 0.0)

        if self.sen_rep_type == 'mp':
            sentence_rep = x.sum(dim=0) / src_lengths.unsqueeze(1)
        else:
            sentence_rep = x[0, :, :]

        if last_state_only:
            inner_states = [x]

        if self.traceable:
            return torch.stack(inner_states), sentence_rep
        else:
            return inner_states, sentence_rep

class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, padding_idx):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx)
        self.reset_parameters(embedding_dim)

    def reset_parameters(self, embedding_dim):
        std = embedding_dim ** -0.5
        nn.init.normal_(self.embed.weight, mean=0, std=std)

    def forward(self, tokens):
        x = self.embed(tokens)
        return x