"""
model.py

Minimal GPT-style Transformer with KV Cache for fast autoregressive inference.

This module implements:

- Token and positional embeddings
- Cached self-attention mechanism
- Transformer blocks with KV caching
- Mini GPT model for text generation

KV caching avoids recomputing attention over the full sequence
during generation, significantly improving inference efficiency.
"""

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    Standard Transformer feed-forward network.
    """

    def __init__(self, dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Embeddings(nn.Module):
    """
    Token + positional embeddings used in the transformer.
    """

    def __init__(self, vocab_size: int, dim: int, max_len: int = 256):
        super().__init__()

        self.token_embeddings = nn.Embedding(vocab_size, dim)
        self.position_embeddings = nn.Embedding(max_len, dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        seq_len = x.shape[1]

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)

        x = self.token_embeddings(x) + self.position_embeddings(positions)

        return self.dropout(x)


class CachedSelfAttention(nn.Module):
    """
    Self-attention layer with KV cache support.

    During generation, previously computed keys and values are stored
    and reused to avoid recomputing attention for the entire sequence.
    """

    def __init__(self, dim: int):

        super().__init__()

        self.qkv = nn.Linear(dim, dim * 3)

        self.out = nn.Linear(dim, dim)

        self.scale = dim ** -0.5

    def forward(self, x: torch.Tensor, cache=None):

        B, T, C = x.shape

        qkv = self.qkv(x)

        q, k, v = qkv.chunk(3, dim=-1)

        if cache is not None:

            k = torch.cat([cache["k"], k], dim=1)

            v = torch.cat([cache["v"], v], dim=1)

        att = (q @ k.transpose(-2, -1)) * self.scale

        att = torch.softmax(att, dim=-1)

        out = att @ v

        new_cache = {"k": k, "v": v}

        return self.out(out), new_cache


class CachedTransformerBlock(nn.Module):
    """
    Transformer block with cached self-attention.
    """

    def __init__(self, dim: int):

        super().__init__()

        self.attn = CachedSelfAttention(dim)

        self.ff = FeedForward(dim)

        self.ln1 = nn.LayerNorm(dim)

        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, cache=None):

        attn_out, new_cache = self.attn(self.ln1(x), cache)

        x = x + attn_out

        x = x + self.ff(self.ln2(x))

        return x, new_cache


class MiniGPTCached(nn.Module):
    """
    Minimal GPT-style language model with KV caching.
    """

    def __init__(self, vocab_size: int, dim: int = 128, layers: int = 4):

        super().__init__()

        self.embed = Embeddings(vocab_size, dim)

        self.blocks = nn.ModuleList(
            [CachedTransformerBlock(dim) for _ in range(layers)]
        )

        self.ln = nn.LayerNorm(dim)

        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x: torch.Tensor, cache=None):

        x = self.embed(x)

        new_caches = []

        for i, block in enumerate(self.blocks):

            layer_cache = None if cache is None else cache[i]

            x, new_cache = block(x, layer_cache)

            new_caches.append(new_cache)

        logits = self.head(self.ln(x))

        return logits, new_caches
