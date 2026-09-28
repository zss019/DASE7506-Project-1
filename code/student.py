"""Student model: a configurable modernised GPT.

Each change relative to the baseline in model.py is a config switch so that
every mechanism can be ablated independently while keeping the same trainer
and scorer:

    norm     'rms' (RMSNorm) | 'layer' (LayerNorm, baseline)
    pos      'rope' (rotary position embedding) | 'learned' (baseline)
    mlp      'swiglu' (gated SiLU MLP) | 'gelu' (baseline)
    mlp_mult hidden width multiplier; SwiGLU defaults to 8/3 so its parameter
             count matches a 4x GELU MLP
    dropout  residual/attention/embedding dropout used only in training
    bias     use biases in linear layers (baseline: True)

With norm='layer', pos='learned', mlp='gelu', dropout=0 and bias=True the
architecture equals the baseline GPT (apart from the scaled init of residual
output projections, controlled by 'scaled_init').
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def make_norm(kind, width):
    return RMSNorm(width) if kind == 'rms' else nn.LayerNorm(width)


class Rotary(nn.Module):
    def __init__(self, head_dim, context, base=10000.):
        super().__init__()
        inv_freq = 1. / base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        angles = torch.outer(torch.arange(context).float(), inv_freq)
        self.register_buffer('cos', angles.cos(), persistent=False)
        self.register_buffer('sin', angles.sin(), persistent=False)

    def forward(self, x):
        # x: [batch, heads, time, head_dim]; rotate channel pairs (i, i + head_dim/2).
        length = x.shape[-2]
        cos, sin = self.cos[:length].to(x.dtype), self.sin[:length].to(x.dtype)
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        width, self.heads = cfg['width'], cfg['heads']
        bias, self.dropout = cfg['bias'], cfg['dropout']
        self.norm1, self.norm2 = make_norm(cfg['norm'], width), make_norm(cfg['norm'], width)
        self.qkv = nn.Linear(width, 3 * width, bias=bias)
        self.proj = nn.Linear(width, width, bias=bias)
        self.proj.residual_out = True
        self.rotary = Rotary(width // self.heads, cfg['context']) if cfg['pos'] == 'rope' else None
        self.gated = cfg['mlp'] == 'swiglu'
        hidden = cfg['mlp_hidden']
        self.up = nn.Linear(width, 2 * hidden if self.gated else hidden, bias=bias)
        self.down = nn.Linear(hidden, width, bias=bias)
        self.down.residual_out = True
        self.drop = nn.Dropout(self.dropout)

    def attention(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).view(batch, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        if self.rotary is not None:
            q, k = self.rotary(q), self.rotary(k)
        attended = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.)
        return self.proj(attended.transpose(1, 2).reshape(batch, length, width))

    def mlp(self, x):
        h = self.up(x)
        if self.gated:
            gate, value = h.chunk(2, dim=-1)
            h = F.silu(gate) * value
        else:
            h = F.gelu(h)
        return self.down(h)

    def forward(self, x):
        x = x + self.drop(self.attention(self.norm1(x)))
        return x + self.drop(self.mlp(self.norm2(x)))


DEFAULTS = dict(norm='rms', pos='rope', mlp='swiglu', mlp_mult=None, dropout=0., bias=False,
                scaled_init=True)


class StudentGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        cfg = DEFAULTS | dict(config)
        if cfg['mlp_mult'] is None:
            cfg['mlp_mult'] = 8 / 3 if cfg['mlp'] == 'swiglu' else 4
        cfg['mlp_hidden'] = int(round(cfg['mlp_mult'] * cfg['width']))
        self.context = cfg['context']
        width = cfg['width']
        self.token = nn.Embedding(cfg['vocab'], width)
        self.pos = nn.Embedding(self.context, width) if cfg['pos'] == 'learned' else None
        self.drop = nn.Dropout(cfg['dropout'])
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg['depth'])])
        self.norm = make_norm(cfg['norm'], width)
        self.head = nn.Linear(width, cfg['vocab'], bias=False)
        residual_std = .02 / math.sqrt(2 * cfg['depth']) if cfg['scaled_init'] else .02
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                std = residual_std if getattr(module, 'residual_out', False) else .02
                nn.init.normal_(module.weight, std=std)
                if getattr(module, 'bias', None) is not None:
                    nn.init.zeros_(module.bias)
        self.head.weight = self.token.weight

    def features(self, ids):
        x = self.token(ids)
        if self.pos is not None:
            x = x + self.pos(torch.arange(ids.shape[1], device=ids.device))
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, ids):
        """Training interface: unnormalized next-token logits [batch, time, vocab]."""
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        """Evaluation interface: stateless, causal, normalized log probabilities."""
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config):
    return StudentGPT(config)
