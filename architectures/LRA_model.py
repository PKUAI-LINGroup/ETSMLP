# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.models import (
    FairseqEncoderModel,
    FairseqEncoder,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    FairseqDropout,
)
# from fairseq.models.lra.transformer_lra_encoder
from  .transformer_lra_encoder import TransformerLRAEncoder
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from ..module.smlp_encoder import SMLPSentenceEncoder

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


@register_model('lra')
class LRAModel(FairseqEncoderModel):
    """
    Class for training a transformer for LRA tasks.
    """
    def __init__(self, args, encoder, task):
        super().__init__(encoder)
        self.encoder = encoder
        self.args = args
        self.use_p = args.use_p
        self._max_positions = args.max_positions
        self.sentence_out_dim = args.sentence_class_num
        self.lm_output_learned_bias = None
        self.dropout_module = FairseqDropout(args.dropout, module_name=self.__class__.__name__)

        self.classifier = nn.ModuleList([])
        if args.classifier_layers > 0:
            self.classifier.append(nn.Sequential(Linear(args.classifier_in_dim, args.classifier_out_dim), self.dropout_module))
            self.classifier.extend([
                nn.Sequential(Linear(args.classifier_out_dim, args.classifier_out_dim), self.dropout_module)
                for _ in range(args.classifier_layers - 1)
            ])
            self.classifier_activation = utils.get_activation_fn(args.classifier_activation_fn)

        self.sentence_projection_layer = Linear(
            args.classifier_out_dim,
            self.sentence_out_dim,
            bias=False
        )
        self.sen_rep_type = getattr(args, "sen_rep_type", "cls")
        self.layer_type = args.layer_type

        # if specified then apply bert initialization on the model. We need
        # to explictly call this to make sure that the output embeddings
        # and projection layers are also correctly initialized
        if getattr(args, 'apply_bert_init', False):
            self.apply(init_bert_params)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # Arguments related to dropout
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float,
                            metavar='D', help='dropout probability for attention weights')
        parser.add_argument('--act-dropout', type=float,
                            metavar='D', help='dropout probability after activation in FFN')
        parser.add_argument('--feature-dropout', action='store_true',
                            help='apply feature dropout')

        # Arguments related to hidden states and self-attention
        parser.add_argument('--encoder-hidden-dim', type=int, metavar='N',
                            help='encoder hidden dimension for Mega')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')

        # Arguments related to input and output embeddings
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--share-encoder-input-output-embed',
                            action='store_true', help='share encoder input and output embeddings')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--no-token-positional-embeddings',
                            action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')

        parser.add_argument('--input-type', choices=['text', 'image'])
        parser.add_argument('--max-positions', type=int,
                            help='number of positional embeddings to learn')
        parser.add_argument('--rel-pos-bias', choices=['simple', 'rotary'], default='simple')

        # Arguments related to sentence level prediction
        parser.add_argument('--sentence-class-num', type=int, metavar='N',
                            help='number of classes for sentence task')
        parser.add_argument('--sent-loss', action='store_true', help='if set, calculate sentence level predictions')

        # Arguments related to parameter initialization
        parser.add_argument('--apply-bert-init', action='store_true',
                            help='use custom param initialization for BERT')

        parser.add_argument('--use-p', default=False, action='store_true',
                            help='use p for prediction')

        # misc params
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--attention-activation-fn', choices=['softmax', 'relu2', 'laplace'],
                            help='activation function for attention mechanism')
        parser.add_argument('--classifier-activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='Which activation function to use for classifier layer.')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')
        parser.add_argument('--quant-noise-pq', type=float, metavar='D', default=0,
                            help='iterative PQ quantization noise at training time')
        parser.add_argument('--quant-noise-pq-block-size', type=int, metavar='D', default=8,
                            help='block size of quantization noise at training time')
        parser.add_argument('--quant-noise-scalar', type=float, metavar='D', default=0,
                            help='scalar quantization noise and scalar quantization at training time')

        parser.add_argument('--layer-type', choices=['transformer', 'luna', 'lstm', 'flash', 'mega'])
        parser.add_argument('--norm-type', choices=['layernorm', 'scalenorm', 'rmsnorm', 'batchnorm', 'syncbatchnorm'])
        parser.add_argument('--normalize-embedding', action='store_true', help='normalize embedding for Mega.')
        parser.add_argument('--sen-rep-type', choices=['cls', 'mp'])

        parser.add_argument('--truncation-length', type=int, metavar='N',
                            help='truncation length of moving average layer.')
        parser.add_argument('--encoder-projection-length', type=int, metavar='N',
                            help='projected length of encoder as key')
        parser.add_argument('--encoder-projected-attention-heads', type=int, metavar='N',
                            help='num encoder projected attention heads')
        parser.add_argument('--decoder-projected-attention-heads', type=int, metavar='N',
                            help='num decoder projected attention heads')

    def forward(self, sample):
        src_tokens = sample['net_input']['src_tokens']
        src_lengths = sample['net_input']['src_lengths']
        sentence_rep = self.encoder(src_tokens, src_lengths)
        if not self.use_p:
            sentence_rep = sentence_rep[1]
        else:
            sentence_rep = sentence_rep[1][1].mean(dim=0)
        if 'net_input1' in sample:
            src1_tokens = sample['net_input1']['src_tokens']
            src1_lengths = sample['net_input1']['src_lengths']
            sentence1_rep = self.encoder(src1_tokens, src1_lengths)
            if not self.use_p:
                if self.layer_type in ['transformer', 'lstm', 'flash', 'mega','smlp']:
                    sentence1_rep = sentence1_rep[1]
                elif self.layer_type == 'luna':
                    sentence1_rep = sentence1_rep[1][0]
            else:
                sentence1_rep = sentence1_rep[1][1].mean(dim=0)
            concat_rep = []
            concat_rep.append(sentence1_rep)
            concat_rep.append(sentence_rep)
            sentence_rep = torch.cat(concat_rep, dim=-1)
        for layer in self.classifier:
            sentence_rep = self.classifier_activation(layer(sentence_rep))
        if self.sentence_projection_layer:
            sentence_logits = self.sentence_projection_layer(sentence_rep)
        return {
            'encoder_out': sentence_logits
        }

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self._max_positions

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        base_architecture(args)
        if not hasattr(args, 'max_positions'):
            args.max_positions = args.tokens_per_sample
        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = args.max_positions
        if not hasattr(args, 'decoder_embed_dim'):
            args.decoder_embed_dim = args.encoder_embed_dim
        encoder = LRAEncoder(args, task)
        return cls(args, encoder, task)


class LRAEncoder(FairseqEncoder):
    """LRA encoder."""

    def __init__(self, args, task):
        if args.input_type == 'text':
            dictionary = task.dictionary
            vocab_size = len(dictionary)
            padding_idx = dictionary.pad_index
            offset_positions_by_padding = True
            embedding_type = 'sparse'
        else:
            assert args.sen_rep_type == 'mp' or args.layer_type == 'lstm'
            dictionary = None
            vocab_size = None
            padding_idx = None
            offset_positions_by_padding = False
            embedding_type = 'linear'
        super().__init__(dictionary)
        self.args = args
        if args.layer_type == 'smlp':
            self.encoder = SMLPSentenceEncoder(args,padding_idx=padding_idx,
                vocab_size=vocab_size,
                num_encoder_layers=args.encoder_layers,
                embedding_dim=args.encoder_embed_dim,
                embedding_type=embedding_type,
                dropout=args.dropout,
                max_seq_len=args.max_positions,
                use_position_embeddings=args.use_position_embeddings,
                offset_positions_by_padding=offset_positions_by_padding,
                encoder_normalize_before=getattr(args, "encoder_normalize_before", False),
                learned_pos_embedding=args.encoder_learned_pos,
                sen_rep_type=getattr(args, 'sen_rep_type', 'cls'),
                freeze=getattr(args,'freeze',False))


    def forward(self, src_tokens, src_lengths=None, **kwargs):
        return self.encoder(src_tokens, src_lengths, last_state_only=True)


@register_model_architecture('lra', 'lra')
def base_architecture(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.act_dropout = getattr(args, 'act_dropout', 0.0)
    args.feature_dropout = getattr(args, 'feature_dropout', False)

    args.encoder_hidden_dim = getattr(args, 'encoder_hidden_dim', 2048)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', args.encoder_hidden_dim)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)

    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.share_encoder_input_output_embed = getattr(args, 'share_encoder_input_output_embed', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', 2048)

    args.sentence_class_num = getattr(args, 'sentence_class_num', 2)
    args.sent_loss = getattr(args, 'sent_loss', True)
    args.apply_bert_init = getattr(args, 'apply_bert_init', True)

    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    args.attention_activation_fn = getattr(args, 'attention_activation_fn', 'relu2')
    args.classifier_layers = getattr(args, 'classifier_layers', 1)
    args.classifier_activation_fn = getattr(args, 'classifier_activation_fn', 'gelu')
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.normalize_embedding = getattr(args, 'normalize_embedding', False)
    args.layer_type = getattr(args, 'layer_type', 'transformer')
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.classifier_in_dim = getattr(args, "classifier_in_dim", args.encoder_embed_dim)
    args.cls_attn = getattr(args, "cls_attn", False)
    args.small_change = getattr(args, "small_change", False)
    args.freeze=getattr(args,'freeze',False)


@register_model_architecture('lra', 'transformer_lra_listop')
def transformer_lra_listop(args):
    args.sentence_class_num = getattr(args, 'sentence_class_num', 10)
    args.max_positions = getattr(args, 'max_positions', 2002)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    args.tie_layer_weights = getattr(args, 'tie_layer_weights', False)
    base_architecture(args)

@register_model_architecture('lra', 'smlp_lra_listop')
def smlp_lra_listop(args):
    args.sentence_class_num = getattr(args, 'sentence_class_num', 10)
    args.max_positions = getattr(args, 'max_positions', 2002)
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 160)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.encoder_q_dim = getattr(args, "encoder_q_dim", 160)
    args.encoder_k_dim = getattr(args, "encoder_k_dim", 160)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.layer_type = getattr(args, 'layer_type', 'smlp')
    args.mha=getattr(args,"mha",False)
    args.only_mha=getattr(args,"only_mha",False)
    args.encoder_hidden_dim = getattr(args, 'encoder_hidden_dim', 160)
    args.sen_rep_type = getattr(args, 'sen_rep_type', 'mp')
    args.classifier_layers = getattr(args, 'classifier_layers', 1)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', args.encoder_embed_dim)
    args.use_position_embeddings = getattr(args, "use_position_embeddings", True)
    args.smlp_pos = getattr(args, 'smlp_pos', 'before_act')
    args.has_ffn = getattr(args, 'has_ffn', False)
    args.kernal_cutoff = getattr(args, 'kernal_cutoff', False)
    args.complex = getattr(args, 'complex', False)
    args.complex_version = getattr(args, 'complex_version', 'normal')
    args.no_beta = getattr(args, 'no_beta', False)
    args.norm_type = getattr(args,'norm_type','layernorm')
    args.max_lambda = getattr(args, "max_lambda", 0.9999)
    args.norm_after_smlp = getattr(args, "norm_after_smlp", False)
    args.gate = getattr(args, 'gate', False)
    args.gate_activation_fn = getattr(args,'gate_activation_fn','sigmoid')
    args.smlp_activation_fn=getattr(args, "smlp_activation_fn", 'relu')
    args.no_omera = getattr(args,"no_omera",False)
    base_architecture(args)


@register_model_architecture('lra', 'smlp_listop_no_pos')
def smlp_lra_listop_1(args):
    args.use_position_embeddings = getattr(args, "use_position_embeddings", False)
    smlp_lra_listop(args)

@register_model_architecture('lra', 'smlp_listop_norm_first')
def smlp_lra_listop_2(args):
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    smlp_lra_listop(args)


@register_model_architecture('lra', 'smlp_listop_tanh')
def smlp_lra_listop_2(args):
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.activation_fn = getattr(args, "activation_fn", "tanh")
    smlp_lra_listop(args)

@register_model_architecture('lra', 'smlp_listop_complex')
def smlp_lra_listop_complex(args):
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 160)
    args.encoder_k_dim = getattr(args, "encoder_k_dim", 160)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.use_position_embeddings = getattr(args, "use_position_embeddings", False)
    args.complex = getattr(args, 'complex', True)
    args.classifier_layers = getattr(args, 'classifier_layers', 0)
    args.r_max=getattr(args, 'r_max', 0.9)
    args.r_min=getattr(args, 'r_min', 0.6)
    args.max_phase=getattr(args, 'max_phase', 6.28)
    args.dt_min=getattr(args, 'dt_min', 1e-3)
    args.dt_max = getattr(args, 'dt_max', 0.1)
    args.cls_attn = getattr(args, "cls_attn", False)
    smlp_lra_listop(args)

@register_model_architecture('lra', 'smlp_listop_complex_cls_attn')
def smlp_lra_listop_complex_cls(args):
    args.cls_attn = getattr(args, "cls_attn", True)
    smlp_lra_listop_complex(args)

@register_model_architecture('lra', 'smlp_listop_complex_test1')
def smlp_lra_listop_complex_test1(args):
    args.r_max=getattr(args, 'r_max', 0.9)
    args.r_min=getattr(args, 'r_min', 0.1)
    args.max_phase=getattr(args, 'max_phase', 3.14)
    smlp_lra_listop_complex(args)

@register_model_architecture('lra', 'smlp_listop_complex_test1_no_omera')
def smlp_lra_listop_complex_no_omera(args):
    args.no_omera = getattr(args,"no_omera",True)
    smlp_lra_listop_complex_test1(args)

@register_model_architecture('lra', 'smlp_listop_complex_cls_attn_test1')
def smlp_lra_listop_complex_cls_test1(args):
    args.cls_attn = getattr(args, "cls_attn", True)
    smlp_lra_listop_complex_test1(args)

@register_model_architecture('lra', 'smlp_listop_complex_test1_2')
def smlp_lra_listop_complex_test2(args):
    args.r_max=getattr(args, 'r_max', 0.6)
    args.r_min=getattr(args, 'r_min', 0.1)
    args.max_phase=getattr(args, 'max_phase', 3.14)
    smlp_lra_listop_complex(args)

@register_model_architecture('lra', 'smlp_listop_complex_test1_3')
def smlp_lra_listop_complex_test2(args):
    args.r_max=getattr(args, 'r_max', 0.5)
    args.r_min=getattr(args, 'r_min', 0.5)
    args.max_phase=getattr(args, 'max_phase', 3.14)
    smlp_lra_listop_complex(args)


@register_model_architecture('lra', 'smlp_listop_complex_test3')
def smlp_lra_listop_complex_test3(args):
    args.r_max=getattr(args, 'r_max', 0.9)
    args.r_min=getattr(args, 'r_min', 0.1)
    args.max_phase=getattr(args, 'max_phase', 3.14)
    args.dt_min=getattr(args, 'dt_min', 0.01)
    args.dt_max = getattr(args, 'dt_max', 1)
    smlp_lra_listop_complex(args)


@register_model_architecture('lra', 'smlp_listop_complex_test4')
def smlp_lra_listop_complex_test4(args):
    args.r_max=getattr(args, 'r_max', 0.9)
    args.r_min=getattr(args, 'r_min', 0.1)
    args.max_phase=getattr(args, 'max_phase', 3.14)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_k_dim = getattr(args, "encoder_k_dim", 256)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    smlp_lra_listop_complex(args)

@register_model_architecture('lra', 'smlp_listop_complex_test5')
def smlp_lra_listop_complex_test5(args):
    args.r_max=getattr(args, 'r_max', 0.9)
    args.r_min=getattr(args, 'r_min', 0.1)
    args.max_phase=getattr(args, 'max_phase', 3.14)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 120)
    args.encoder_k_dim = getattr(args, "encoder_k_dim", 120)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    smlp_lra_listop_complex(args)

@register_model_architecture('lra', 'smlp_listop_norm_smlp')
def smlp_lra_listop_norm_smlp(args):
    args.norm_after_smlp = getattr(args, "norm_after_smlp", True)
    smlp_lra_listop_complex(args)

@register_model_architecture('lra', 'smlp_listop_complex_theta')
def smlp_lra_listop_complex_theta(args):
    args.complex_version = getattr(args, 'complex_version', 'lambda_theta')
    smlp_lra_listop_complex(args)

@register_model_architecture('lra', 'smlp_listop_complex_normalize')
def smlp_listop_complex_normalize(args):
    args.complex_version = getattr(args, 'complex_version', 'normalize')
    smlp_lra_listop_complex(args)

@register_model_architecture('lra', 'smlp_listop_complex_new_normal')
def smlp_listop_complex_new_normal(args):
    args.complex_version = getattr(args, 'complex_version', 'new_normal')
    smlp_lra_listop_complex_test1(args)

@register_model_architecture('lra', 'smlp_listop_complex_h')
def smlp_listop_complex_normal_h(args):
    args.complex_version = getattr(args, 'complex_version', 'normal_h')
    args.complex_h = getattr(args,"complex_h",64)
    smlp_lra_listop_complex_test1(args)

@register_model_architecture('lra', 'smlp_listop_complex_no_alpha')
def ssmlp_listop_complex_out_alpha(args):
    args.complex_version = getattr(args, 'complex_version', 'no_alpha')
    smlp_lra_listop_complex_test1(args)

@register_model_architecture('lra', 'smlp_listop_complex_stable_5')
def ssmlp_listop_complex_stable(args):
    args.complex_version = getattr(args, 'complex_version', 'stable')
    args.w_r = 0.5
    args.w_i = 0
    smlp_lra_listop_complex_test1(args)

@register_model_architecture('lra', 'smlp_listop_complex_stable_1')
def ssmlp_listop_complex_stable1(args):
    args.complex_version = getattr(args, 'complex_version', 'stable')
    args.w_r = 0.1
    args.w_i = 0
    smlp_lra_listop_complex_test1(args)

@register_model_architecture('lra', 'smlp_listop_complex_stable_3')
def ssmlp_listop_complex_stable1(args):
    args.complex_version = getattr(args, 'complex_version', 'stable')
    args.w_r = 0.3
    args.w_i = 0
    smlp_lra_listop_complex_test1(args)

@register_model_architecture('lra', 'smlp_listop_complex_stable_7')
def ssmlp_listop_complex_stable1(args):
    args.complex_version = getattr(args, 'complex_version', 'stable')
    args.w_r = 0.7
    args.w_i = 0
    smlp_lra_listop_complex_test1(args)

@register_model_architecture('lra', 'smlp_listop_complex_stable_9')
def ssmlp_listop_complex_stable1(args):
    args.complex_version = getattr(args, 'complex_version', 'stable')
    args.w_r = 0.9
    args.w_i = 0
    smlp_lra_listop_complex_test1(args)

@register_model_architecture('lra', 'smlp_listop_complex_gelu')
def ssmlp_listop_complex_gelu(args):
    args.smlp_activation_fn=getattr(args, "smlp_activation_fn", 'gelu')
    smlp_lra_listop_complex_test1(args)

@register_model_architecture('lra', 'smlp_listop_complex_gate')
def ssmlp_listop_complex_gate(args):
    args.gate = getattr(args, 'gate', True)
    smlp_lra_listop_complex_test1(args)

@register_model_architecture('lra', 'smlp_listop_complex_gate_gleu_another')
def ssmlp_listop_complex_gate_gleu(args):
    args.gate_activation_fn = getattr(args,'gate_activation_fn','gleu')
    args.gate = getattr(args, 'gate', True)
    smlp_lra_listop_complex_test1(args)

@register_model_architecture('lra', 'smlp_listop_complex_ml')
def ssmlp_listop_complex_ml(args):
    args.max_lambda = getattr(args, "max_lambda", 1)
    smlp_lra_listop_complex_test1(args)

@register_model_architecture('lra', 'smlp_listop_complex_no_beta')
def smlp_lra_listop_new_1(args):
    args.no_beta = getattr(args, 'no_beta', True)
    smlp_lra_listop_complex(args)


@register_model_architecture('lra', 'smlp_listop_after_act')
def smlp_lra_listop_2(args):
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.smlp_pos = getattr(args, 'smlp_pos', 'after_act')
    smlp_lra_listop(args)

@register_model_architecture('lra', 'smlp_listop_no_class')
def smlp_lra_listop_2(args):
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.classifier_layers = getattr(args, 'classifier_layers', 0)
    smlp_lra_listop(args)

@register_model_architecture('lra', 'smlp_listop_before_in')
def smlp_lra_listop_2(args):
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.smlp_pos = getattr(args, 'smlp_pos', 'before_in')
    smlp_lra_listop(args)

@register_model_architecture('lra', 'smlp_listop_after_out')
def smlp_lra_listop_2(args):
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.smlp_pos = getattr(args, 'smlp_pos', 'after_out')
    smlp_lra_listop(args)

@register_model_architecture('lra', 'smlp_listop_ffn')
def smlp_lra_listop_2(args):
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.has_ffn = getattr(args, 'has_ffn', True)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    smlp_lra_listop(args)

#Text
@register_model_architecture('lra', 'transformer_lra_imdb')
def transformer_lra_imdb_architecture(args):
    args.max_positions = getattr(args, 'max_positions', 4002)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    args.encoder_layers = getattr(args, 'encoder_layers', 4)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.classifier_layers = getattr(args, 'classifier_layers', 1)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', 1024)
    base_architecture(args)

@register_model_architecture('lra', 'smlp_lra_imdb_complex')
def smlp_lra_imdb_complex(args):
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 160)
    args.encoder_k_dim = getattr(args, "encoder_k_dim", 160)
    args.encoder_layers = getattr(args, "encoder_layers", 4)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', 160)
    args.use_position_embeddings = getattr(args, "use_position_embeddings", False)
    args.complex = getattr(args, 'complex', True)
    args.classifier_layers = getattr(args, 'classifier_layers', 0)
    args.sentence_class_num = getattr(args, 'sentence_class_num', 2)
    args.truncation_length = getattr(args, 'truncation_length', 1024)
    args.max_positions = getattr(args, 'max_positions', 4002)
    args.r_max=getattr(args, 'r_max', 0.999)
    args.r_min=getattr(args, 'r_min', 0.9)
    args.max_phase=getattr(args, 'max_phase', 0.314)
    args.dt_min=getattr(args, 'dt_min', 1e-3)
    args.dt_max = getattr(args, 'dt_max', 0.1)
    smlp_lra_listop(args)

@register_model_architecture('lra', 'smlp_lra_imdb_complex_test1')
def smlp_lra_imdb_complex_test1(args):
    args.r_max=getattr(args, 'r_max', 0.9)
    args.r_min=getattr(args, 'r_min', 0.1)
    args.max_phase=getattr(args, 'max_phase', 3.14)
    args.dt_min=getattr(args, 'dt_min', 1e-3)
    args.dt_max = getattr(args, 'dt_max', 0.1)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    smlp_lra_imdb_complex(args)

@register_model_architecture('lra', 'smlp_lra_imdb_complex_test2')
def smlp_lra_imdb_complex_test2(args):
    args.r_max=getattr(args, 'r_max', 0.9)
    args.r_min=getattr(args, 'r_min', 0.6)
    args.max_phase=getattr(args, 'max_phase', 3.14)
    args.dt_min=getattr(args, 'dt_min', 1e-3)
    args.dt_max = getattr(args, 'dt_max', 0.1)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    smlp_lra_imdb_complex(args)

@register_model_architecture('lra', 'smlp_lra_imdb_complex_test3')
def smlp_lra_imdb_complex_test3(args):
    args.r_max=getattr(args, 'r_max', 0.9)
    args.r_min=getattr(args, 'r_min', 0.1)
    args.max_phase=getattr(args, 'max_phase', 3.14)
    args.dt_min=getattr(args, 'dt_min', 1e-3)
    args.dt_max = getattr(args, 'dt_max', 0.1)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_k_dim = getattr(args, "encoder_k_dim", 256)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', 256)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    smlp_lra_imdb_complex(args)

@register_model_architecture('lra', 'smlp_lra_imdb_complex_test4')
def smlp_lra_imdb_complex_test4(args):
    args.r_max=getattr(args, 'r_max', 0.9)
    args.r_min=getattr(args, 'r_min', 0.1)
    args.max_phase=getattr(args, 'max_phase', 3.14)
    args.dt_min=getattr(args, 'dt_min', 1e-3)
    args.dt_max = getattr(args, 'dt_max', 0.1)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 160)
    args.encoder_k_dim = getattr(args, "encoder_k_dim", 160)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    smlp_lra_imdb_complex(args)

@register_model_architecture('lra', 'smlp_lra_imdb_complex_gate_gleu')
def smlp_lra_imdb_complex_gate_gleu(args):
    args.gate_activation_fn = getattr(args,'gate_activation_fn','gleu')
    args.gate = getattr(args, 'gate', True)
    smlp_lra_imdb_complex_test4(args)

@register_model_architecture('lra', 'smlp_lra_imdb_complex_gate')
def smlp_lra_imdb_complex_gate_gleu(args):
    # args.gate_activation_fn = getattr(args,'gate_activation_fn','gleu')
    args.gate = getattr(args, 'gate', True)
    smlp_lra_imdb_complex_test1(args)


#Retrieval
@register_model_architecture('lra', 'transformer_lra_aan')
def transformer_lra_aan_architecture(args):
    args.apply_bert_init = getattr(args, 'apply_bert_init', False)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', 4)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.classifier_layers = getattr(args, 'classifier_layers', 1)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', 512)
    args.classifier_in_dim = getattr(args, 'classifier_in_dim', args.encoder_embed_dim * 2)
    base_architecture(args)

@register_model_architecture('lra', 'smlp_lra_aan_complex')
def smlp_lra_aan_complex(args):
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 160)
    args.encoder_k_dim = getattr(args, "encoder_k_dim", 160)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', 160)
    args.classifier_in_dim = getattr(args, 'classifier_in_dim', args.encoder_embed_dim * 2)
    args.use_position_embeddings = getattr(args, "use_position_embeddings", False)
    args.complex = getattr(args, 'complex', True)
    args.classifier_layers = getattr(args, 'classifier_layers', 1)
    args.sentence_class_num = getattr(args, 'sentence_class_num', 2)
    args.truncation_length = getattr(args, 'truncation_length', 1024)
    args.max_positions = getattr(args, 'max_positions', 4002)
    args.r_max=getattr(args, 'r_max', 0.9)
    args.r_min=getattr(args, 'r_min', 0.1)
    args.max_phase=getattr(args, 'max_phase', 3.14)
    args.dt_min=getattr(args, 'dt_min', 1e-3)
    args.dt_max = getattr(args, 'dt_max', 0.1)
    smlp_lra_listop(args)

@register_model_architecture('lra', 'smlp_lra_aan_complex_test1')
def smlp_lra_ann_complex_test1(args):
    args.r_max=getattr(args, 'r_max', 0.99)
    args.r_min=getattr(args, 'r_min', 0.9)
    args.max_phase=getattr(args, 'max_phase', 0.314)
    args.dt_min=getattr(args, 'dt_min', 1e-3)
    args.dt_max = getattr(args, 'dt_max', 0.1)
    smlp_lra_aan_complex(args)

@register_model_architecture('lra', 'smlp_lra_aan_complex_test2')
def smlp_lra_ann_complex_test2(args):
    args.r_max=getattr(args, 'r_max', 0.99)
    args.r_min=getattr(args, 'r_min', 0.9)
    args.max_phase=getattr(args, 'max_phase', 3.14)
    smlp_lra_aan_complex(args)

@register_model_architecture('lra', 'smlp_lra_aan_complex_test3')
def smlp_lra_ann_complex_test3(args):
    args.r_max=getattr(args, 'r_max', 0.9)
    args.r_min=getattr(args, 'r_min', 0.1)
    args.max_phase=getattr(args, 'max_phase', 0.314)
    smlp_lra_aan_complex(args)

@register_model_architecture('lra', 'smlp_lra_aan_complex_gate')
def smlp_lra_aan_complex_gate_gleu(args):
    args.gate = getattr(args, 'gate', True)
    smlp_lra_ann_complex_test1(args)

#Image
@register_model_architecture('lra', 'transformer_lra_cifar10')
def transformer_lra_cifar10(args):
    args.apply_bert_init = getattr(args, 'apply_bert_init', False)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 128)
    args.encoder_layers = getattr(args, 'encoder_layers', 1)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 64)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.classifier_layers = getattr(args, 'classifier_layers', 1)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', 128)
    args.sentence_class_num = getattr(args, 'sentence_class_num', 10)
    args.max_positions = getattr(args, 'max_positions', 1024)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    base_architecture(args)

@register_model_architecture('lra', 'smlp_lra_cifar10_complex')
def smlp_lra_cifar10_complex(args):
    args.norm_type = getattr(args, 'norm_type', 'batchnorm')
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 128)
    args.encoder_k_dim = getattr(args, "encoder_k_dim", 256)
    args.encoder_layers = getattr(args, "encoder_layers", 8)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', 128)
    args.use_position_embeddings = getattr(args, "use_position_embeddings", False)
    args.complex = getattr(args, 'complex', True)
    args.classifier_layers = getattr(args, 'classifier_layers', 0)
    args.truncation_length = getattr(args, 'truncation_length', 1024)
    args.max_positions = getattr(args, 'max_positions', 1024)
    args.r_max=getattr(args, 'r_max', 0.99)
    args.r_min=getattr(args, 'r_min', 0.9)
    args.max_phase=getattr(args, 'max_phase', 3.14)
    args.dt_min=getattr(args, 'dt_min', 1e-3)
    args.dt_max = getattr(args, 'dt_max', 0.1)
    args.cls_attn = getattr(args,'cls_attn',False)
    smlp_lra_listop(args)

@register_model_architecture('lra', 'smlp_lra_cifar10_complex_pn')
def smlp_lra_cifar10_complex_pn(args):
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    smlp_lra_cifar10_complex(args)

@register_model_architecture('lra', 'smlp_lra_cifar10_complex_gate')
def smlp_lra_cifar10_complex_gate(args):
    args.r_max=getattr(args, 'r_max', 0.9)
    args.r_min=getattr(args, 'r_min', 0.6)
    args.max_phase=getattr(args, 'max_phase', 3.14)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 160)
    args.encoder_k_dim = getattr(args, "encoder_k_dim", 320)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', 160)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.gate = getattr(args, 'gate', True)
    smlp_lra_cifar10_complex(args)

@register_model_architecture('lra', 'smlp_lra_cifar10_complex_gate_gleu')
def ssmlp_lra_cifar10_complex_gate_gleu(args):
    args.gate_activation_fn = getattr(args,'gate_activation_fn','gleu')
    args.gate = getattr(args, 'gate', True)
    smlp_lra_cifar10_complex_gate(args)


@register_model_architecture('lra', 'smlp_lra_cifar10_complex_cl')
def smlp_lra_cifar10_complex_cl(args):
    args.r_max=getattr(args, 'r_max', 0.9)
    args.r_min=getattr(args, 'r_min', 0.1)
    args.max_phase=getattr(args, 'max_phase', 3.14)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 160)
    args.encoder_k_dim = getattr(args, "encoder_k_dim", 320)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', 320)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.classifier_layers = getattr(args, 'classifier_layers', 1)
    args.classifier_in_dim = getattr(args, 'classifier_in_dim', 160)
    smlp_lra_cifar10_complex(args)

@register_model_architecture('lra', 'smlp_lra_cifar10_complex_test1')
def smlp_lra_cifar10_complex_test1(args):
    args.r_max=getattr(args, 'r_max', 0.9)
    args.r_min=getattr(args, 'r_min', 0.6)
    args.max_phase=getattr(args, 'max_phase', 3.14)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 160)
    args.encoder_k_dim = getattr(args, "encoder_k_dim", 320)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', 160)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    smlp_lra_cifar10_complex(args)


@register_model_architecture('lra', 'smlp_lra_cifar10_complex_test3')
def smlp_lra_cifar10_complex_test3(args):
    args.r_max=getattr(args, 'r_max', 0.99)
    args.r_min=getattr(args, 'r_min', 0.9)
    args.max_phase=getattr(args, 'max_phase', 3.14)
    smlp_lra_cifar10_complex_test1(args)

# pathfinder
@register_model_architecture('lra', 'transformer_lra_pf32')
def transformer_lra_pf32(args):
    args.apply_bert_init = getattr(args, 'apply_bert_init', False)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 128)
    args.encoder_layers = getattr(args, 'encoder_layers', 1)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.classifier_layers = getattr(args, 'classifier_layers', 1)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', 256)
    args.sentence_class_num = getattr(args, 'sentence_class_num', 2)
    args.max_positions = getattr(args, 'max_positions', 1026)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    args.sen_rep_type = getattr(args, 'sen_rep_type', 'mp')
    base_architecture(args)



@register_model_architecture('lra', 'smlp_lra_complex_pf32')
def smlp_lra_pf32_complex(args):
    args.norm_type = getattr(args, 'norm_type', 'batchnorm')
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 128)
    args.encoder_k_dim = getattr(args, "encoder_k_dim", 256)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', 128)
    args.use_position_embeddings = getattr(args, "use_position_embeddings", False)
    args.complex = getattr(args, 'complex', True)
    args.classifier_layers = getattr(args, 'classifier_layers', 0)
    args.sentence_class_num = getattr(args, 'sentence_class_num', 2)
    args.truncation_length = getattr(args, 'truncation_length', 1024)
    args.max_positions = getattr(args, 'max_positions', 1024)
    args.r_max=getattr(args, 'r_max', 0.9999)
    args.r_min=getattr(args, 'r_min', 0.9)
    args.max_phase=getattr(args, 'max_phase', 0.314)
    args.dt_min=getattr(args, 'dt_min', 1e-3)
    args.dt_max = getattr(args, 'dt_max', 0.1)
    args.cls_attn = getattr(args,'cls_attn',False)
    smlp_lra_listop(args)

@register_model_architecture('lra', 'smlp_lra_complex_pf32_test1')
def smlp_lra_pf32_complex_test1(args):
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.cls_attn = getattr(args,'cls_attn',True)
    smlp_lra_pf32_complex(args)

@register_model_architecture('lra', 'smlp_lra_complex_pf32_gate')
def smlp_lra_pf32_complex_gate(args):
    args.gate = getattr(args, 'gate', True)
    smlp_lra_pf32_complex(args)

@register_model_architecture('lra', 'smlp_lra_complex_pf32_new_normal')
def smlp_lra_pf32_complex_new_normal(args):
    args.complex_version = getattr(args, 'complex_version', 'new_normal')
    smlp_lra_pf32_complex(args)

@register_model_architecture('lra', 'smlp_lra_complex_pf32_gate_gelu')
def smlp_lra_complex_pf32_gate_gelu(args):
    args.gate_activation_fn = getattr(args,'gate_activation_fn','gelu')
    smlp_lra_pf32_complex_gate(args)


@register_model_architecture('lra', 'smlp_lra_complex_pf32_lambda2')
def smlp_lra_pf32_complex_1(args):
    args.r_max=getattr(args, 'r_max', 0.99)
    args.max_lambda = getattr(args, "max_lambda", 0.99)
    smlp_lra_pf32_complex(args)

@register_model_architecture('lra', 'smlp_lra_complex_pf32_lambda1')
def smlp_lra_pf32_complex_1(args):
    args.max_lambda = getattr(args, "max_lambda", 1)
    smlp_lra_pf32_complex(args)

@register_model_architecture('lra', 'smlp_lra_complex_pf32_h')
def smlp_lra_pf32_complex_1(args):
    args.complex_version = getattr(args, 'complex_version', 'normal_h')
    args.complex_h = getattr(args, 'complex_h', 64)
    smlp_lra_pf32_complex(args)

@register_model_architecture('lra', 'smlp_lra_complex_pf32_theta')
def smlp_lra_pf32_complex_base(args):
    args.complex_version = getattr(args, 'complex_version', 'lambda_theta')
    smlp_lra_pf32_complex(args)

@register_model_architecture('lra', 'smlp_lra_complex_pf32_sigmoid')
def smlp_lra_pf32_complex_base(args):
    args.complex_version = getattr(args, 'complex_version', 'sigmoid')
    smlp_lra_pf32_complex(args)

@register_model_architecture('lra', 'smlp_lra_complex_pf32_another_init')
def smlp_lra_pf32_complex_base(args):
    args.r_max=getattr(args, 'r_max', 0.9)
    args.r_min=getattr(args, 'r_min', 0.6)
    args.max_phase=getattr(args, 'max_phase', 6.28)
    args.dt_min=getattr(args, 'dt_min', 1e-3)
    args.dt_max = getattr(args, 'dt_max', 0.1)
    args.no_beta = getattr(args, 'no_beta', True)
    smlp_lra_pf32_complex(args)

@register_model_architecture('lra', 'smlp_lra_complex_pf32_normalize')
def smlp_lra_pf32_complex_base(args):
    args.r_max=getattr(args, 'r_max', 0.999)
    args.r_min=getattr(args, 'r_min', 0.9)
    args.max_phase=getattr(args, 'max_phase', 0.628)
    args.dt_min=getattr(args, 'dt_min', 1e-3)
    args.dt_max = getattr(args, 'dt_max', 0.1)
    args.no_beta = getattr(args, 'no_beta', True)
    args.smlp_pos = getattr(args, 'smlp_pos', 'before_in')
    args.complex_version = getattr(args, 'complex_version', 'normalize')
    smlp_lra_pf32_complex(args)

@register_model_architecture('lra', 'smlp_lra_complex_pf32_norm_smlp')
def smlp_lra_listop_norm_smlp(args):
    args.norm_after_smlp = getattr(args, "norm_after_smlp", True)
    smlp_lra_pf32_complex(args)

@register_model_architecture('lra', 'smlp_lra_complex_pf32_small_change')
def smlp_lra_pf32_small_change(args):
    args.small_change = getattr(args, "small_change", True)
    smlp_lra_pf32_complex(args)

@register_model_architecture('lra', 'smlp_lra_complex_pf32_base')
def smlp_lra_pf32_complex_base(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 128)
    args.encoder_k_dim = getattr(args, "encoder_k_dim", 256)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', 128)
    smlp_lra_pf32_complex(args)

@register_model_architecture('lra', 'smlp_lra_complex_pf32_class')
def smlp_lra_pf32_complex_base(args):
    args.classifier_layers = getattr(args, 'classifier_layers', 1)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', 256)
    smlp_lra_pf32_complex(args)


@register_model_architecture('lra', 'smlp_lra_complex_pf128')
def smlp_lra_pf128_complex(args):
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 128)
    args.encoder_k_dim = getattr(args, "encoder_k_dim", 128)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', 128)
    args.use_position_embeddings = getattr(args, "use_position_embeddings", False)
    args.complex = getattr(args, 'complex', True)
    args.classifier_layers = getattr(args, 'classifier_layers', 0)
    args.sentence_class_num = getattr(args, 'sentence_class_num', 2)
    args.truncation_length = getattr(args, 'truncation_length', 4096)
    args.max_positions = getattr(args, 'max_positions', 128 * 128)
    args.r_max=getattr(args, 'r_max', 0.9999)
    args.r_min=getattr(args, 'r_min', 0.9)
    args.max_lambda = getattr(args, "max_lambda", 0.9999)
    args.max_phase=getattr(args, 'max_phase', 0.314)
    args.dt_min=getattr(args, 'dt_min', 1e-3)
    args.dt_max = getattr(args, 'dt_max', 0.1)
    smlp_lra_pf32_complex(args)

@register_model_architecture('lra', 'smlp_lra_complex_pf128_base')
def smlp_lra_pf128_complex_test1(args):
    args.r_max=getattr(args, 'r_max', 0.9999)
    args.r_min=getattr(args, 'r_min', 0.99)
    args.encoder_k_dim = getattr(args, "encoder_k_dim", 256)
    args.max_phase=getattr(args, 'max_phase', 0.314)
    args.dt_min=getattr(args, 'dt_min', 1e-3)
    args.dt_max = getattr(args, 'dt_max', 0.1)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.small_change = getattr(args,"small_change",False)
    smlp_lra_pf128_complex(args)

@register_model_architecture('lra', 'smlp_lra_complex_pf128_test2')
def smlp_lra_pf128_complex_test2(args):
    args.small_change = True
    args.r_max=getattr(args, 'r_max', 0.9999)
    args.r_min=getattr(args, 'r_min', 0.999)
    args.dt_min=getattr(args, 'dt_min', 1e-3)
    args.dt_max = getattr(args, 'dt_max', 0.1)
    args.max_phase=getattr(args, 'max_phase', 0.314)
    smlp_lra_pf128_complex_test1(args)

@register_model_architecture('lra', 'smlp_lra_complex_pf128_freeze')
def smlp_lra_pf128_complex_test2(args):
    args.small_change = True
    args.freeze = True
    args.r_max=getattr(args, 'r_max', 0.999)
    args.r_min=getattr(args, 'r_min', 0.9)
    args.dt_min=getattr(args, 'dt_min', 1e-3)
    args.dt_max = getattr(args, 'dt_max', 0.1)
    args.max_phase=getattr(args, 'max_phase', 0.314)
    smlp_lra_pf128_complex_test1(args)

@register_model_architecture('lra', 'smlp_lra_complex_pf128_gate')
def smlp_lra_pf128_complex_gate(args):
    args.r_max=getattr(args, 'r_max', 0.9999)
    args.r_min=getattr(args, 'r_min', 0.99)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 128)
    args.encoder_k_dim = getattr(args, "encoder_k_dim", 256)
    args.encoder_layers = getattr(args, "encoder_layers",6)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', 128)
    args.gate = getattr(args, 'gate', True)
    smlp_lra_pf128_complex(args)