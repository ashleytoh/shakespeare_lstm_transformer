#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer Language Model
Character-level Transformer for text generation
"""

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """Single Transformer block with multi-head attention and feed-forward network"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.2):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = self.ln1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.ln2(x + self.dropout(ff_out))
        return x


class TransformerLanguageModel(nn.Module):
    """Transformer-based language model for character-level text generation"""
    
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout=0.2, max_len=2048):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.embed_dropout = nn.Dropout(dropout * 0.5)  # Embedding dropout
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
        # Pre-compute and cache mask for common sequence lengths
        self.register_buffer("mask_cache", None, persistent=False)
        self._precompute_mask(max_len)

    def _precompute_mask(self, max_len):
        """Pre-compute causal mask for efficiency"""
        mask = torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def _causal_mask(self, L, device):
        """Get causal mask for sequence length L"""
        return self.causal_mask[:L, :L].to(device)

    def forward(self, x):
        B, T = x.shape
        device = x.device
        # Pre-compute position embeddings
        pos = torch.arange(T, device=device).unsqueeze(0)
        h = self.tok(x) + self.pos(pos)
        h = self.embed_dropout(h)  # Add embedding dropout
        mask = self._causal_mask(T, device)
        for block in self.blocks:
            h = block(h, attn_mask=mask)
        h = self.ln_f(h)
        return self.head(h)
