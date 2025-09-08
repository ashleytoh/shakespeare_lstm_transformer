#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM Language Model
Character-level LSTM for text generation
"""

import torch
import torch.nn as nn

class LSTMLanguageModel(nn.Module):
    """
    LSTM-based language model for character-level text generation
    
    Args:
        vocab_size: Size of the vocabulary (number of unique characters)
        embed_size: Embedding dimension
        hidden_size: LSTM hidden state dimension  
        num_layers: Number of LSTM layers
        dropout: Dropout probability for regularization
    """
    def __init__(self, vocab_size, embed_size=256, hidden_size=256, num_layers=1, dropout=0.2):
        super().__init__()
        
        # Character embedding layer
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # Input dropout
        self.drop_in = nn.Dropout(dropout)
        
        # LSTM layers with dropout between them
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        
        # Output dropout
        self.drop_out = nn.Dropout(dropout)
        
        # Final linear layer to vocabulary
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # Store dimensions for reference
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x, hidden=None):
        """
        Forward pass through the LSTM model
        
        Args:
            x: Input token indices [batch_size, seq_len]
            hidden: Previous hidden state (optional)
            
        Returns:
            logits: Predictions over vocabulary [batch_size, seq_len, vocab_size]
            hidden: Updated hidden state
        """
        # Embed input tokens
        x = self.embed(x)  # [batch_size, seq_len, embed_size]
        
        # Apply input dropout
        x = self.drop_in(x)
        
        # Pass through LSTM
        y, hidden = self.lstm(x, hidden)  # [batch_size, seq_len, hidden_size]
        
        # Apply output dropout
        y = self.drop_out(y)
        
        # Project to vocabulary size
        logits = self.fc(y)  # [batch_size, seq_len, vocab_size]
        
        return logits, hidden
    
    def init_hidden(self, batch_size, device='cpu'):
        """
        Initialize hidden state for the LSTM
        
        Args:
            batch_size: Batch size
            device: Device to create tensors on
            
        Returns:
            hidden: Initialized hidden state (h_0, c_0)
        """
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h_0, c_0)
