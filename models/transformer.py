#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer Language Model
Character-level Transformer for text generation
"""

import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    """
    Single Transformer block with multi-head attention and feed-forward network
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward network dimension
        dropout: Dropout probability
    """
    def __init__(self, d_model=256, n_heads=4, d_ff=1024, dropout=0.2):
        super().__init__()
        
        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # Layer normalization after attention
        self.ln1 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # GELU activation function
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        
        # Layer normalization after feed-forward
        self.ln2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        """
        Forward pass through transformer block
        
        Args:
            x: Input embeddings [batch_size, seq_len, d_model]
            attn_mask: Attention mask to prevent looking ahead
            
        Returns:
            x: Output embeddings [batch_size, seq_len, d_model]
        """
        # Multi-head attention with residual connection
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = self.ln1(x + self.drop(attn_out))
        
        # Feed-forward with residual connection
        ff_out = self.ff(x)
        x = self.ln2(x + self.drop(ff_out))
        
        return x

class TinyTransformerLM(nn.Module):
    """
    Tiny Transformer Language Model for character-level text generation
    
    Args:
        vocab_size: Size of the vocabulary
        d_model: Model dimension (embedding size)
        n_layers: Number of transformer layers
        n_heads: Number of attention heads per layer
        d_ff: Feed-forward network dimension
        dropout: Dropout probability
        max_len: Maximum sequence length for positional embeddings
    """
    def __init__(self, vocab_size, d_model=256, n_layers=3, n_heads=4, d_ff=1024, dropout=0.2, max_len=2048):
        super().__init__()
        
        # Token embeddings
        self.tok = nn.Embedding(vocab_size, d_model)
        
        # Positional embeddings
        self.pos = nn.Embedding(max_len, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_layers)
        ])
        
        # Final layer normalization
        self.ln_f = nn.LayerNorm(d_model)
        
        # Output head to vocabulary
        self.head = nn.Linear(d_model, vocab_size)
        
        # Cache for causal attention mask
        self.register_buffer("mask_cache", None, persistent=False)
        
        # Store configuration
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_len = max_len

    def _causal_mask(self, L, device):
        """
        Create causal attention mask to prevent looking ahead
        
        Args:
            L: Sequence length
            device: Device to create mask on
            
        Returns:
            mask: Causal mask [L, L]
        """
        if self.mask_cache is None or self.mask_cache.size(0) < L:
            # Create upper triangular matrix (True values will be masked)
            m = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
            self.mask_cache = m
        return self.mask_cache[:L, :L].to(device)

    def forward(self, x):
        """
        Forward pass through the transformer
        
        Args:
            x: Input token indices [batch_size, seq_len]
            
        Returns:
            logits: Predictions over vocabulary [batch_size, seq_len, vocab_size]
        """
        B, T = x.shape
        device = x.device
        
        # Create position indices
        pos = torch.arange(T, device=device).unsqueeze(0)  # [1, T]
        
        # Token + positional embeddings
        h = self.tok(x) + self.pos(pos)  # [batch_size, seq_len, d_model]
        
        # Create causal mask
        mask = self._causal_mask(T, device)
        
        # Pass through transformer blocks
        for blk in self.blocks:
            h = blk(h, attn_mask=mask)
        
        # Final layer norm
        h = self.ln_f(h)
        
        # Project to vocabulary
        return self.head(h)  # [batch_size, seq_len, vocab_size]
