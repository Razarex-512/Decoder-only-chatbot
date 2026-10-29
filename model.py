"""Small, explicit decoder-only transformer; no pretrained components."""
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    vocab_size: int
    context: int = 256
    width: int = 96
    layers: int = 2
    heads: int = 4
    dropout: float = 0.1
    pad_id: int = 0
    position_jitter: bool = False

    def __post_init__(self):
        if min(self.vocab_size, self.context, self.width, self.layers, self.heads) < 1:
            raise ValueError("Model dimensions must be positive.")
        if self.width % self.heads or not 0 <= self.dropout < 1:
            raise ValueError("Width must divide into heads; dropout must be in [0, 1).")


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.width // config.heads
        self.query = nn.Linear(config.width, config.width)
        self.key = nn.Linear(config.width, config.width)
        self.value = nn.Linear(config.width, config.width)
        self.projection = nn.Linear(config.width, config.width)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer("causal", torch.ones(config.context, config.context,
                                                  dtype=torch.bool).tril())

    def forward(self, x, valid, inspect=False):
        batch, length, width = x.shape
        def heads(projection):
            return projection(x).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = heads(self.query), heads(self.key), heads(self.value)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)  # [B,H,T,T]
        allowed = self.causal[:length, :length][None, None] & valid[:, None, None, :]
        # A padded query with no earlier valid key gets a harmless self edge.
        # Valid queries always have their own key. Padding never contributes to loss.
        empty = ~allowed.any(dim=-1, keepdim=True)
        eye = torch.eye(length, dtype=torch.bool, device=x.device)[None, None]
        allowed = allowed | (empty & eye)
        weights = scores.masked_fill(~allowed, float("-inf")).softmax(dim=-1)
        output = (self.dropout(weights) @ v).transpose(1, 2).contiguous().view(batch, length, width)
        return self.projection(output), weights if inspect else None


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.width)
        self.norm2 = nn.LayerNorm(config.width)
        self.attention = Attention(config)
        self.feed_forward = nn.Sequential(nn.Linear(config.width, 4 * config.width),
                                          nn.GELU(), nn.Linear(4 * config.width, config.width))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, valid, inspect=False):
        attended, weights = self.attention(self.norm1(x), valid, inspect)
        x = x + self.dropout(attended)
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x, weights


class TinyLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.width)
        self.position_embedding = nn.Embedding(config.context, config.width)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.layers)])
        self.final_norm = nn.LayerNorm(config.width)
        self.output = nn.Linear(config.width, config.vocab_size, bias=False)
        self.apply(self._initialize)
        self.output.weight = self.token_embedding.weight  # One unique trainable matrix.

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, tokens, inspect=False):
        if tokens.ndim != 2 or not 1 <= tokens.shape[1] <= self.config.context:
            raise ValueError("Expected token IDs [B,T] with 1 <= T <= context.")
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        if self.training and self.config.position_jitter:
            # Shift each whole example's position IDs without changing token order.
            # This exposes answers at varied absolute positions instead of letting
            # the tiny corpus teach a fixed answer-start position shortcut.
            offsets = torch.randint(self.config.context - tokens.shape[1] + 1,
                                    (tokens.shape[0], 1), device=tokens.device)
            positions = positions + offsets
        x = self.token_embedding(tokens) + self.position_embedding(positions)  # [B,T,C]
        valid = tokens.ne(self.config.pad_id)
        attention = [] if inspect else None
        for block in self.blocks:
            x, weights = block(x, valid, inspect)
            if inspect:
                attention.append(weights)
        return self.output(self.final_norm(x)), attention  # [B,T,V]


def next_token_loss(logits, targets):
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                           ignore_index=0)  # [B*T,V], [B*T]; PAD is always ID 0.
