#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Factory for Shakespeare Language Models
Creates LSTM and Transformer models with standardized interface
"""

from .lstm_model import LSTMLanguageModel
from .transformer_model import TransformerLanguageModel


def create_model(model_type, vocab_size, embed_size=128, hidden_size=128, 
                num_layers=2, dropout=0.1, n_heads=4, ff_size=512, max_len=512):
    """
    Create model based on configuration with optional parameter overrides
    
    Args:
        model_type: "lstm" or "transformer"
        vocab_size: Size of vocabulary
        embed_size: Embedding dimension
        hidden_size: Hidden dimension (LSTM) / d_model (Transformer)
        num_layers: Number of layers
        dropout: Dropout probability
        n_heads: Number of attention heads (Transformer only)
        ff_size: Feed-forward network dimension (Transformer only)
        max_len: Maximum sequence length (Transformer only)
    
    Returns:
        PyTorch model instance
    """
    if model_type == "lstm":
        return LSTMLanguageModel(
            vocab_size=vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
    elif model_type == "transformer":
        return TransformerLanguageModel(
            vocab_size=vocab_size,
            d_model=embed_size,
            n_layers=num_layers,
            n_heads=n_heads,
            d_ff=ff_size,
            dropout=dropout,
            max_len=max_len
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Must be 'lstm' or 'transformer'.")
