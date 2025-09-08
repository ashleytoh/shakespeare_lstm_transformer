#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shakespeare Language Model - Complete Training and Generation Pipeline
Created based on DSA4213 Assignment 2

A comprehensive script that trains LSTM/Transformer models on Shakespeare text
and demonstrates text generation capabilities.

@author: DSA4213 Student
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import time
import math
import matplotlib
matplotlib.use("Agg")  # headless plots
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION - Set all hyperparameters here
# =============================================================================

# Data Configuration
DATA_PATH = "data/input.txt"                   # Shakespeare text corpus
OUTPUT_DIR = "reports"                         # Output directory for models/plots

# Model Selection
MODEL_TYPE = "lstm"                            # "lstm" or "transformer"

# Data Processing
SEQUENCE_LENGTH = 256                          # Sequence length for training
BATCH_SIZE = 64                               # Batch size
TRAIN_SPLIT = 0.8                             # Training data fraction
VAL_SPLIT = 0.1                               # Validation data fraction (test = 1-train-val)

# Model Architecture
EMBED_SIZE = 256                              # Embedding dimension
HIDDEN_SIZE = 256                             # Hidden size (LSTM) / d_model (Transformer)
NUM_LAYERS = 2                                # Number of layers
DROPOUT = 0.2                                 # Dropout probability

# Transformer-specific (ignored for LSTM)
N_HEADS = 4                                   # Number of attention heads
FF_SIZE = 1024                                # Feed-forward dimension

# Training Configuration
EPOCHS = 10                                   # Number of training epochs
LEARNING_RATE = 1e-3                          # Adam learning rate
GRADIENT_CLIP = 1.0                           # Gradient clipping threshold
SEED = 42                                     # Random seed

# Generation Configuration
GENERATION_PROMPTS = ["HAMLET:", "To be or not to be", "Romeo"]
GENERATION_LENGTH = 200                       # Number of tokens to generate
TEMPERATURES = [0.7, 1.0, 1.3]              # Temperature values for sampling

# =============================================================================

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_shakespeare_text():
    """Load and preview the Shakespeare text corpus"""
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    
    print("=" * 60)
    print("SHAKESPEARE TEXT CORPUS")
    print("=" * 60)
    print("First 500 characters:")
    print(text[:500])
    print("...")
    print(f"Total characters: {len(text):,}")
    
    return text

def build_vocabulary(text):
    """Build character-level vocabulary"""
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    # Character to index mapping
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    
    print(f"Vocabulary size: {vocab_size} unique characters")
    print(f"Vocabulary preview: {chars[:20]}...")
    
    return char_to_idx, idx_to_char, vocab_size

def encode_text(text, char_to_idx):
    """Convert text to numerical indices"""
    return [char_to_idx[char] for char in text]

class ShakespeareDataset(Dataset):
    """Character-level Shakespeare dataset"""
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return x, y

class LSTMLanguageModel(nn.Module):
    """LSTM-based language model"""
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, hidden=None):
        x = self.embedding(x)
        output, hidden = self.lstm(x, hidden)
        output = self.dropout(output)
        logits = self.fc(output)
        return logits, hidden

class TransformerBlock(nn.Module):
    """Single Transformer block"""
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
    """Transformer-based language model"""
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout=0.2, max_len=2048):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.register_buffer("mask_cache", None, persistent=False)

    def _causal_mask(self, L, device):
        if self.mask_cache is None or self.mask_cache.size(0) < L:
            m = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
            self.mask_cache = m
        return self.mask_cache[:L, :L].to(device)

    def forward(self, x):
        B, T = x.shape
        device = x.device
        pos = torch.arange(T, device=device).unsqueeze(0)
        h = self.tok(x) + self.pos(pos)
        mask = self._causal_mask(T, device)
        for block in self.blocks:
            h = block(h, attn_mask=mask)
        h = self.ln_f(h)
        return self.head(h)

def create_model(model_type, vocab_size):
    """Create model based on configuration"""
    if model_type == "lstm":
        return LSTMLanguageModel(
            vocab_size=vocab_size,
            embed_size=EMBED_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT
        )
    elif model_type == "transformer":
        return TransformerLanguageModel(
            vocab_size=vocab_size,
            d_model=EMBED_SIZE,
            n_layers=NUM_LAYERS,
            n_heads=N_HEADS,
            d_ff=FF_SIZE,
            dropout=DROPOUT,
            max_len=SEQUENCE_LENGTH
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def train_model(model, train_loader, val_loader, device):
    """Training loop"""
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {MODEL_TYPE.upper()}")
    print(f"Parameters: {total_params:,}")
    print(f"Device: {device}")
    print()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        total_train_loss = 0
        train_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for x_batch, y_batch in progress_bar:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            # Initialize hidden state for LSTM
            if MODEL_TYPE == "lstm":
                hidden = None
            
            optimizer.zero_grad()
            
            # Forward pass
            if MODEL_TYPE == "lstm":
                output, hidden = model(x_batch, hidden)
            else:
                output = model(x_batch)
            
            # Calculate loss
            loss = criterion(output.view(-1, output.size(-1)), y_batch.view(-1))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            
            total_train_loss += loss.item()
            train_batches += 1
            progress_bar.set_postfix(loss=loss.item())
        
        # Validation
        model.eval()
        total_val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                
                if MODEL_TYPE == "lstm":
                    output, _ = model(x_batch, None)
                else:
                    output = model(x_batch)
                
                loss = criterion(output.view(-1, output.size(-1)), y_batch.view(-1))
                total_val_loss += loss.item()
                val_batches += 1
        
        # Calculate average losses
        avg_train_loss = total_train_loss / train_batches
        avg_val_loss = total_val_loss / val_batches
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'char_to_idx': char_to_idx,
                'idx_to_char': idx_to_char,
                'vocab_size': vocab_size,
                'config': {
                    'model_type': MODEL_TYPE,
                    'embed_size': EMBED_SIZE,
                    'hidden_size': HIDDEN_SIZE,
                    'num_layers': NUM_LAYERS,
                    'dropout': DROPOUT,
                    'n_heads': N_HEADS,
                    'ff_size': FF_SIZE,
                    'seq_len': SEQUENCE_LENGTH
                }
            }, os.path.join(OUTPUT_DIR, f'best_{MODEL_TYPE}_model.pt'))
    
    return train_losses, val_losses

def plot_training_curves(train_losses, val_losses):
    """Plot and save training curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.title(f'{MODEL_TYPE.upper()} Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, f'{MODEL_TYPE}_training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {OUTPUT_DIR}/{MODEL_TYPE}_training_curves.png")

def generate_text(model, start_text, max_len=200, temperature=1.0, device='cpu'):
    """Generate text using the trained model"""
    model.eval()
    
    # Encode start text
    input_ids = torch.tensor([char_to_idx.get(char, 0) for char in start_text], 
                            dtype=torch.long).unsqueeze(0).to(device)
    generated_text = list(start_text)
    hidden = None
    
    with torch.no_grad():
        for _ in range(max_len):
            if MODEL_TYPE == "lstm":
                output, hidden = model(input_ids[:, -1:], hidden)
                output = output[:, -1, :]
            else:
                output = model(input_ids)
                output = output[:, -1, :]
            
            # Apply temperature
            output = output / temperature
            probabilities = F.softmax(output, dim=-1)
            
            # Sample next character
            next_char_id = torch.multinomial(probabilities, num_samples=1).item()
            next_char = idx_to_char[next_char_id]
            
            generated_text.append(next_char)
            input_ids = torch.tensor([[next_char_id]], dtype=torch.long).to(device)
    
    return ''.join(generated_text)

def demonstrate_generation(model, device):
    """Demonstrate text generation with different prompts and temperatures"""
    print("\n" + "=" * 60)
    print("TEXT GENERATION DEMO")
    print("=" * 60)
    
    for prompt in GENERATION_PROMPTS:
        print(f"\nStarting prompt: '{prompt}'")
        print("-" * 40)
        
        for temp in TEMPERATURES:
            print(f"\nTemperature {temp}:")
            generated = generate_text(model, prompt, GENERATION_LENGTH, temp, device)
            print(generated[:300] + "..." if len(generated) > 300 else generated)

def evaluate_model(model, test_loader, device):
    """Evaluate model on test set"""
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    total_batches = 0
    
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            if MODEL_TYPE == "lstm":
                output, _ = model(x_batch, None)
            else:
                output = model(x_batch)
            
            loss = criterion(output.view(-1, output.size(-1)), y_batch.view(-1))
            total_loss += loss.item()
            total_batches += 1
    
    avg_loss = total_loss / total_batches
    perplexity = math.exp(avg_loss)
    
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Perplexity: {perplexity:.2f}")
    
    return avg_loss, perplexity

def main():
    """Main function - complete pipeline"""
    print("SHAKESPEARE LANGUAGE MODEL TRAINING")
    print("Configuration:")
    print(f"  Model: {MODEL_TYPE}")
    print(f"  Sequence Length: {SEQUENCE_LENGTH}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    
    # Set seed for reproducibility
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    
    # Load and process data
    text = load_shakespeare_text()
    global char_to_idx, idx_to_char, vocab_size  # Make global for generation functions
    char_to_idx, idx_to_char, vocab_size = build_vocabulary(text)
    data = encode_text(text, char_to_idx)
    
    # Create train/val/test splits
    n_train = int(len(data) * TRAIN_SPLIT)
    n_val = int(len(data) * VAL_SPLIT)
    
    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]
    
    print(f"\nData splits:")
    print(f"  Train: {len(train_data):,} characters")
    print(f"  Validation: {len(val_data):,} characters")
    print(f"  Test: {len(test_data):,} characters")
    
    # Create datasets and dataloaders
    train_dataset = ShakespeareDataset(train_data, SEQUENCE_LENGTH)
    val_dataset = ShakespeareDataset(val_data, SEQUENCE_LENGTH)
    test_dataset = ShakespeareDataset(test_data, SEQUENCE_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Create and train model
    model = create_model(MODEL_TYPE, vocab_size).to(device)
    train_losses, val_losses = train_model(model, train_loader, val_loader, device)
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses)
    
    # Load best model for evaluation and generation
    checkpoint = torch.load(os.path.join(OUTPUT_DIR, f'best_{MODEL_TYPE}_model.pt'), 
                           map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate model
    test_loss, test_perplexity = evaluate_model(model, test_loader, device)
    
    # Demonstrate text generation
    demonstrate_generation(model, device)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Best model saved to: {OUTPUT_DIR}/best_{MODEL_TYPE}_model.pt")
    print(f"Final test perplexity: {test_perplexity:.2f}")

if __name__ == "__main__":
    main()
