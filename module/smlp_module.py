from typing import Dict, List, Optional,Tuple
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from fairseq import utils
from torch import Tensor
import torch.fft
from .sequence_norm import SequenceNorm
import math

_c2r = torch.view_as_real
_r2c = torch.view_as_complex

EPS=1e-4
class SMLP_module(nn.Module):
    """SMLP module
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        num_heads,
        q_dim,
        k_dim,
        args,
        casual=False,
        bias=True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.activation_fn = utils.get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )
        assert args.smlp_pos in ['before_in','before_act','after_act','after_out']
        self.smlp_pos=args.smlp_pos
        self.norm_after_smlp = args.norm_after_smlp
        # self.in_head_dim = k_dim // num_heads
        self.gate = args.gate
        self.casual = casual

        # self.q_proj = nn.Linear(input_dim, q_dim, bias=bias)
        self.args=args
        self.k_proj = nn.Linear(input_dim, k_dim, bias=bias)
        if self.gate:
            # self.g_proj = nn.Linear(input_dim, k_dim, bias=bias)
            self.g_proj = nn.Linear(input_dim, output_dim, bias=bias)
            self.g_act = utils.get_activation_fn(
                                activation=str(args.gate_activation_fn)
                                if getattr(args, "args.gate_activation_fn", None) is not None
                                else "relu"
                            ) if args.gate_activation_fn !='sigmoid' else torch.sigmoid
            # self.another_proj = nn.Linear(output_dim, output_dim, bias=bias)
        # ema_dim=k_dim
        if self.casual:
            bidirectional = False
        else:
            bidirectional = True

        # else:
        if self.smlp_pos=='before_in':
            self.cets = CETS(input_dim,bidirectional=bidirectional,args=args)
            norm_dim=input_dim
        elif self.smlp_pos=='after_out':
            self.cets = CETS(output_dim,bidirectional=bidirectional,args=args)
            norm_dim=output_dim
        else:
            self.cets = CETS(k_dim,bidirectional=bidirectional,args=args)
            norm_dim = k_dim

        self.out_proj = nn.Linear(k_dim, output_dim, bias=bias)
        # if self.gate:
        #     self.out_out_proj = nn.Linear(output_dim, output_dim, bias=bias)
        if args.norm_after_smlp:
            self.smlp_norm = SequenceNorm(args.norm_type, norm_dim)

        self.reset_parameters()


    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.gate:
            nn.init.xavier_uniform_(self.g_proj.weight)


    def forward(
        self,
        query,
        key_padding_mask: Optional[Tensor] = None,
        extra_position:Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.input_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        # q
        # q = self.q_proj(query)
        # q = self.activation_fn(q)

        if self.smlp_pos == 'before_in':
            query,_ = self.cets(query,padding_mask=key_padding_mask)
            if self.norm_after_smlp:
                query = self.smlp_norm(query)

        #MHSD
        k = self.k_proj(query)

        if self.gate:
            # g = self.g_proj(query)
            g = self.g_act(self.g_proj(query))
        # v = self.v_proj(query)

        assert k is not None
        src_len = k.size(0)
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len


        if key_padding_mask is not None:
            k = k.masked_fill(key_padding_mask.transpose(0, 1).unsqueeze(2).to(torch.bool), 0)


        if self.smlp_pos=='before_act':
            k,_ = self.cets(k,padding_mask=key_padding_mask)
            if self.norm_after_smlp:
                k = self.smlp_norm(k)

        qz = k

        # attn = torch.cat((q,qz1),axis=-1)
        attn = self.activation_fn(qz)

        if self.smlp_pos=='after_act':
            attn,_ = self.cets(attn,padding_mask=key_padding_mask)
            if self.norm_after_smlp:
                attn = self.smlp_norm(attn)

        # # Try another gate
        # if self.gate:
        #     attn = g*attn

        attn = self.out_proj(attn)

        if self.smlp_pos=='after_out':
            attn,_ = self.cets(attn,padding_mask=key_padding_mask)
            if self.norm_after_smlp:
                attn = self.smlp_norm(attn)

        if self.gate:
            attn = g*attn

            # attn = self.another_proj(attn)

        return attn


class CETS(nn.Module):
    def __init__(self,dim,bidirectional=False,args=None):
        super().__init__()
        self.bidirectional=bidirectional
        # self.bidirectional = False
        kernel_dim = 2 * dim if self.bidirectional else dim
        self.dim=dim
        self.complex = args.complex
        self.max_q = args.max_lambda
        self.cls_attn = args.cls_attn
        self.args=args
        self.small_change =args.small_change if hasattr(args,"small_change") else False
        self.no_omera = args.no_omera if hasattr(args,"no_omera") else False
        if self.complex:
            self.r_max=args.r_max
            self.r_min=args.r_min
            self.max_phase=args.max_phase
            self.dt_min=args.dt_min
            self.dt_max = args.dt_max
            self.version = args.complex_version
            if self.complex and (self.version == "lambda_theta" or self.version == "sigmoid"):
                self.log_lambda = nn.Parameter(torch.Tensor(kernel_dim))
                self.theta = nn.Parameter(torch.Tensor(kernel_dim))
                # self.alpha = nn.Parameter(torch.Tensor(kernel_dim, 2))
                self.no_beta = args.no_beta
                if not self.no_beta:
                    self.beta = nn.Parameter(torch.Tensor(kernel_dim, 2))
            elif self.complex and (self.version=='normal' or self.version == 'normalize' or self.version == 'no_alpha' or self.version == 'new_normal' or self.version=='stable'):
                self.log_delta = nn.Parameter(torch.Tensor(kernel_dim,2))
                if self.version!='no_alpha':
                    self.alpha = nn.Parameter(torch.Tensor(kernel_dim, 2))
                self.no_beta = args.no_beta
                if not self.no_beta:
                    self.beta = nn.Parameter(torch.Tensor(kernel_dim, 2))
            elif self.complex and self.version=='normal_h':
                self.complex_h = args.complex_h
                self.log_delta = nn.Parameter(torch.Tensor(self.complex_h,2))
                self.alpha = nn.Parameter(torch.Tensor(self.complex_h, 2))
                self.beta = nn.Parameter(torch.Tensor(kernel_dim,self.complex_h, 2))
            else:
                self.delta = nn.Parameter(torch.Tensor(kernel_dim, 2))
                self.alpha = nn.Parameter(torch.Tensor(kernel_dim, 2))
                self.no_beta = args.no_beta

                if not self.no_beta:
                    self.beta = nn.Parameter(torch.Tensor(kernel_dim, 2))

            # if self.version == 'normalize':
            #     self.h0 = nn.Parameter(torch.Tensor(kernel_dim))
        else:
            self.delta = nn.Parameter(torch.Tensor(kernel_dim, 1))
            self.alpha = nn.Parameter(torch.Tensor(kernel_dim, 1))
            self.beta = nn.Parameter(torch.Tensor(kernel_dim, 1))
        if not self.no_omera:
            self.omega = nn.Parameter(torch.Tensor(dim))

        self.reset_parameters()

    def reset_parameters(self):

        with torch.no_grad():
            # delta & alpha
            if self.complex and (self.version == 'normal' or self.version == 'normalize' or self.version == 'no_alpha'):

                N=self.log_delta.size(0)
                u1 = torch.rand(N,dtype = torch.float)
                u2 = torch.rand(N,dtype = torch.float)
                nu_log = torch.log(-1*0.5*torch.log(u1*(self.r_max**2-self.r_min**2) + self.r_min**2))
                theta_log = self.max_phase*u2
                w = -1*nu_log.exp()+1j*theta_log
                self.log_delta = nn.Parameter(_c2r(w.to(torch.cfloat).log()))
                if self.version!='no_alpha':
                    self.alpha =nn.Parameter( (math.log(self.dt_min) + torch.rand(N) * (math.log(self.dt_max) - math.log(self.dt_min))).unsqueeze(-1).tile(2))
                # self.beta = nn.Parameter(_c2r(torch.log(torch.sqrt(1-(_r2c(self.log_delta).exp()*_r2c(self.alpha.exp())).exp()**2).abs().to(w))))
                if not self.no_beta:
                    # self.beta = nn.Parameter(_c2r(torch.log(torch.sqrt(1-(_r2c(self.log_delta*self.alpha.exp()).exp()**2).abs())).to(w)))
                    # self.beta = nn.Parameter(_c2r(torch.log(torch.sqrt(1-(_r2c(self.log_delta).exp()).exp().abs()**2)).to(w)))
                    nn.init.normal_(self.beta, mean=0.0, std=0.2)



            elif self.complex and self.version=='stable':

                N=self.log_delta.size(0)
                assert hasattr(self.args,'w_r') and hasattr(self.args,'w_i')
                w = (torch.ones(N)*(self.args.w_r+1j * self.args.w_i)).log()
                self.log_delta = nn.Parameter(_c2r(w.to(torch.cfloat).log()))
                self.alpha =nn.Parameter( (math.log(self.dt_min) + torch.rand(N) * (math.log(self.dt_max) - math.log(self.dt_min))).unsqueeze(-1).tile(2))
                nn.init.normal_(self.beta, mean=0.0, std=0.2)

            elif self.complex and self.version=='normal_h':
                N=self.log_delta.size(0)
                nu_log = -0.5*torch.ones(N)
                theta_log = math.pi * torch.arange(N)
                self.log_delta[:,0] = nu_log
                self.log_delta[:,1] = theta_log
                self.alpha =nn.Parameter( (math.log(self.dt_min) + torch.rand(N) * (math.log(self.dt_max) - math.log(self.dt_min))).unsqueeze(-1).tile(2))
                nn.init.normal_(self.beta, mean=0.0, std=0.2)

            elif self.complex and self.version == 'new_normal':
                N=self.log_delta.size(0)
                nu_log = -0.5*torch.ones(N)
                theta_log = math.pi * torch.arange(N)
                self.log_delta[:,0] = nu_log
                self.log_delta[:,1] = theta_log
                self.alpha =nn.Parameter( (math.log(self.dt_min) + torch.rand(1) * (math.log(self.dt_max) - math.log(self.dt_min))).unsqueeze(-1).tile(2))
                nn.init.normal_(self.beta, mean=0.0, std=0.2)

            elif self.complex and self.version == 'lambda_theta':

                r_max=self.r_max
                r_min=self.r_min
                max_phase=self.max_phase
                # dt_min=1e-3
                # dt_max = 0.1
                N=self.log_lambda.size(0)
                u1 = torch.rand(N,dtype = torch.float)
                u2 = torch.rand(N,dtype = torch.float)
                nu_log = torch.log(-1*(0.5*torch.log(u1*(r_max**2-r_min**2) + r_min**2)))
                theta = max_phase*u2
                w = -1*nu_log.exp()+1j*theta
                self.log_lambda = nn.Parameter(nu_log)
                self.theta = nn.Parameter(theta.log())
                # self.alpha =nn.Parameter( (math.log(dt_min) + torch.rand(N) * (math.log(dt_max) - math.log(dt_min))).unsqueeze(-1).tile(2))
                if not self.no_beta:
                    self.beta = nn.Parameter(_c2r(torch.log(torch.sqrt(1-(w.exp()).abs()**2)).to(w)))

            else:
                nn.init.normal_(self.delta, mean=1, std=0.2)
                nn.init.normal_(self.alpha, mean=1, std=0.2)

                # beta
                if not self.no_beta:
                    nn.init.normal_(self.beta, mean=0.0, std=0.2)
            # if self.small_change:
            #     self.beta=self.beta.double()
            #     self.alpha = self.alpha.double()
            #     self.log_lambda = self.log_lambda.double()
            # omega
            if not self.no_omera:
                nn.init.normal_(self.omega, mean=0.0, std=1.0)


    def complex_kernal(self,sq):

        # alpha = torch.sigmoid(self.alpha)
        # p_num ,p_theta = sigmoid_complex(self.delta*alpha)
        # q = p_num.log()+1j*p_theta
        if self.version == 'normal' or self.version == 'normalize' or self.version=='stable':
            #
            q = _r2c(self.log_delta).exp()*_r2c(self.alpha.exp())
            # constraint function
            q = q+(q.exp().abs()>self.max_q)*(self.max_q/q.exp().abs()).log()

            if self.small_change:
                q = q*0.06

            vander = torch.arange(sq).to(self.log_delta).view(1, sq) * q.unsqueeze(-1)



        elif self.version == 'no_alpha':
            q = _r2c(self.log_delta).exp()
            q = q+(q.exp().abs()>self.max_q)*(self.max_q/q.exp().abs()).log()
            vander = torch.arange(sq).to(self.log_delta).view(1, sq) * q.unsqueeze(-1)


        else:
            raise NotImplementedError

        vander_exp = vander.exp()

        if self.no_beta:
            k = ((1-q.exp())).unsqueeze(-1)*vander_exp
        else:
            # exp to out
            if self.small_change:

                k = (_r2c(self.beta).exp()*(1-q.exp())*torch.sqrt(1-torch.square(q.exp().abs()))).unsqueeze(-1)*vander_exp

            else:
                k = (_r2c(self.beta).exp()*(1-q.exp())).unsqueeze(-1)*vander_exp

        return k.float()


    def kernal(self,sq):

        p= torch.sigmoid(self.delta)
        # p = 2*torch.sigmoid(self.delta)

        alpha = torch.sigmoid(self.alpha)
        q = 1- p * alpha
        # q = p
        #vandermonde
        vander = torch.arange(sq).to(p).view(1, sq) * torch.log(q)
        vander_exp = torch.exp(vander)

        # delta*(1-a*delta)^k
        k = (p*self.beta) * vander_exp
        # kernel = (1-p)*self.beta * vander_exp

        return k


    def forward(self,x,padding_mask=None):
        # x: [sequence_length,batch_size,hidden_states]
        #fft
        sq = x.size(0)
        if not self.no_omera:
            residual = x * self.omega
        # L x B x D -> B x D x L
        x = x.permute(1, 2, 0)
        # padding
        if padding_mask is not None:
            x = x * (1.0 - padding_mask.unsqueeze(1).type_as(x))


        xc = x

        if self.complex:
            kernel = self.complex_kernal(sq)
        else:
            kernel = self.kernal(sq)

        # kernel =  self.gamma*kernel
        if self.bidirectional:
            k1, k2 = torch.split(kernel, [self.dim, self.dim], dim=0)
            # old method
            # # D x 2*L-1
            # kernel = F.pad(k1, (sq - 1, 0)) + F.pad(k2.flip(-1), (0, sq - 1))
            # x = F.pad(x, (sq - 1, 0))
            # fft_len = 2*sq - 1
            # s = 2 * sq - 2
            # dss method
            # D x 2*L-1
            kernel = F.pad(k1, (0, sq)) + F.pad(k2.flip(-1), (sq,0))
            # x = F.pad(x, (sq - 1, 0))
        fft_len = sq
        s=0

        k_f = torch.fft.rfft(kernel.float(), n=2 * fft_len)
        x_f = torch.fft.rfft(xc.float(), n=2 * fft_len)
        # B x D x L
        out = torch.fft.irfft(x_f * k_f, n=2 * fft_len)[..., s:s + sq]
        if self.version == 'normalize':
            if self.bidirectional:
                pass
            else:
                out = out/torch.cumsum(kernel,dim=1).unsqueeze(0)
        out = out.type_as(x)
        if self.no_omera:
            out = out.permute(2, 0, 1)
        else:
            out = out.permute(2, 0, 1) + residual
        # out = out.permute(2, 0, 1) + residual
        # out = F.relu(out.permute(2, 0, 1))
        z = out
        #EMA
        # z=self.EMA(k,padding_mask)
        return z,padding_mask


