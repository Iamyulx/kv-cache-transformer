# KV Cache Transformer – Fast Autoregressive Inference

This project implements a Transformer with Key-Value (KV) Cache in PyTorch to demonstrate how modern large language models accelerate autoregressive text generation.

The KV cache avoids recomputing attention for the entire sequence at every step, significantly improving inference efficiency.

This technique is used in modern models such as:

GPT‑4

LLaMA

PaLM



# Motivation

During standard autoregressive generation, the transformer recomputes attention over the entire sequence every time a new token is generated.

Example:

Input:

Cybersecurity is

Generation process without caching:

Step 1 → compute attention for tokens [Cybersecurity, is]
Step 2 → recompute attention for tokens [Cybersecurity, is, next_token]
Step 3 → recompute attention again
Step 4 → recompute again

This leads to quadratic computational cost.



# Solution: KV Cache

Instead of recomputing keys and values at every step, we store them in a cache and reuse them.

Generation with KV cache:

Step 1 → compute attention normally
Step 2 → reuse previous K,V + compute only new token
Step 3 → reuse again
Step 4 → reuse again

This reduces computation to linear complexity during generation.




# Architecture

The model implemented in this repository is a minimal GPT-style transformer.

Components:

Token Embeddings
      ↓
Position Embeddings
      ↓
Transformer Blocks (with KV Cache)
      ↓
LayerNorm
      ↓
Linear Language Modeling Head

Each Transformer block contains:

LayerNorm
   ↓
Cached Self Attention
   ↓
Residual Connection
   ↓
Feed Forward Network
   ↓
Residual Connection




# Cached Self Attention

The key idea is to store previous keys and values and append the new ones.

if cache is not None:

    k = torch.cat([cache["k"],k],dim=1)

    v = torch.cat([cache["v"],v],dim=1)

This allows the model to reuse previous attention computations.




# Model Implementation

Main components:

FeedForward Network
class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )


        
## Cached Self Attention

Stores previous keys and values during generation.

class CachedSelfAttention(nn.Module):

    def forward(self, x, cache=None):

        qkv = self.qkv(x)

        q,k,v = qkv.chunk(3,dim=-1)

        if cache is not None:
            k = torch.cat([cache["k"],k],dim=1)
            v = torch.cat([cache["v"],v],dim=1)



## Transformer Block
class CachedTransformerBlock(nn.Module):

    def forward(self,x,cache=None):

        attn_out, new_cache = self.attn(self.ln1(x),cache)

        x = x + attn_out

        x = x + self.ff(self.ln2(x))

        return x, new_cache



# Fast Text Generation

The generate_fast function demonstrates cached autoregressive generation.

def generate_fast(model, start, tokenizer, steps=100):

    tokens = tokenizer.encode(start)

    x = torch.tensor(tokens).unsqueeze(0)

    cache = None

    for _ in range(steps):

        logits, cache = model(x[:,-1:], cache)

        next_token = torch.argmax(logits[:,-1,:],dim=-1)

        x = torch.cat([x,next_token.unsqueeze(0)],dim=1)

    return tokenizer.decode(x[0].tolist())


# Example Result
with caching: 0.1964 seconds

Even with a small toy model, caching significantly improves inference speed.

In large models (billions of parameters), KV caching is essential for real-time generation.



# Project Structure
kv-cache-transformer/

│
├── model.py
│   Transformer with KV Cache
│
├── generate.py
│   Fast autoregressive generation
│
├── tokenizer.py
│   Simple tokenizer for testing
│
├── benchmark.py
│   Speed comparison
│
├── requirements.txt
│
└── README.md


# Installation

Clone the repository:

git clone https://github.com/Iamyulx/kv-cache-transformer

Install dependencies:

pip install -r requirements.txt


# Requirements

torch



# Learning Objectives

This project demonstrates:

Transformer architecture

Autoregressive generation

Attention caching

Efficient inference in LLMs

PyTorch model design



# Future Improvements

Possible extensions:

Multi-Head Attention

Flash Attention

Beam Search

Transformer scaling experiments

HuggingFace integration
