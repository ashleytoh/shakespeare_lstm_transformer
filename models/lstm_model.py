#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM Language Model
Character-level LSTM for text generation
"""

import torch
import torch.nn as nn


class LSTMLanguageModel(nn.Module):
    """LSTM-based language model for character-level text generation"""
    
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Add embedding dropout for better regularization
        self.embed_dropout = nn.Dropout(dropout * 0.5)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
    def forward(self, x, hidden=None):
        x = self.embedding(x)
        x = self.embed_dropout(x)  # Add embedding dropout
        output, hidden = self.lstm(x, hidden)
        output = self.dropout(output)
        logits = self.fc(output)
        return logits, hidden
