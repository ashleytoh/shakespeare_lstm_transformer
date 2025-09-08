#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shakespeare Language Model Training
Trains LSTM and Transformer models on Shakespeare text corpus
"""

import os, time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import set_seed, load_text, build_vocab_char, encode, split_indices, make_loader, plot_curves, save_ckpt
from models.lstm import LSTMLanguageModel
from models.transformer import TinyTransformerLM

# =============================================================================
# HYPERPARAMETERS - Configure your training setup here
# =============================================================================

# Data and Model Configuration
DATA_PATH = "data/input.txt"            # Path to Shakespeare text corpus
MODEL_TYPE = "lstm"                     # "lstm" or "transformer" 
SEQUENCE_LENGTH = 256                   # Max sequence length for training
BATCH_SIZE = 64                         # Training batch size

# Model Architecture
EMBED_SIZE = 256                        # Embedding dimension
HIDDEN_SIZE = 256                       # Hidden dimension (LSTM) / d_model (Transformer)
NUM_LAYERS = 2                          # Number of layers
DROPOUT = 0.2                           # Dropout probability

# Transformer-specific parameters (ignored for LSTM)
N_HEADS = 4                             # Number of attention heads
FF_SIZE = 1024                          # Feed-forward dimension

# Training Configuration
EPOCHS = 10                             # Number of training epochs
LEARNING_RATE = 1e-3                    # Adam learning rate
GRADIENT_CLIP = 1.0                     # Gradient clipping threshold
SEED = 42                               # Random seed for reproducibility

# Output Configuration
OUTPUT_DIR = "reports"                  # Directory for saving models and plots

# =============================================================================

def get_model(name, vocab_size):
    """Create model instance based on configuration"""
    if name == "lstm":
        return LSTMLanguageModel(
            vocab_size=vocab_size,
            embed_size=EMBED_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT
        )
    elif name == "transformer":
        return TinyTransformerLM(
            vocab_size=vocab_size,
            d_model=EMBED_SIZE,
            n_layers=NUM_LAYERS,
            n_heads=N_HEADS,
            d_ff=FF_SIZE,
            dropout=DROPOUT,
            max_len=SEQUENCE_LENGTH
        )
    else:
        raise ValueError("Unknown model name")

def run_epoch(model, loader, criterion, optimizer, device, clip=None, train=True):
    """Run one training/validation epoch"""
    if train: 
        model.train()
    else: 
        model.eval()
        
    total_loss, n_batches = 0.0, 0
    
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            
            if train: 
                optimizer.zero_grad()
                
            # Forward pass - handle different model types
            if hasattr(model, "lstm"):
                logits, _ = model(x, None)
            else:
                logits = model(x)
                
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            
            if train:
                loss.backward()
                if clip: 
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                
            total_loss += float(loss.detach().cpu())
            n_batches += 1
            
    return total_loss / max(n_batches, 1)

def main():
    """Main training function"""
    # Set random seed for reproducibility
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Training {MODEL_TYPE} model on Shakespeare corpus...")

    # Load and preprocess data
    print(f"Loading text from {DATA_PATH}...")
    text = load_text(DATA_PATH)
    stoi, itos = build_vocab_char(text)
    ids = encode(text, stoi)
    
    print(f"Dataset size: {len(ids)} characters")
    print(f"Vocabulary size: {len(stoi)} unique characters")

    # Create train/val/test splits (80/10/10)
    (s0, e0), (s1, e1), (s2, e2) = split_indices(len(ids))
    train_ids, val_ids, test_ids = ids[s0:e0], ids[s1:e1], ids[s2:e2]

    # Create data loaders
    train_loader = make_loader(train_ids, SEQUENCE_LENGTH, BATCH_SIZE, shuffle=True, device=device)
    val_loader   = make_loader(val_ids, SEQUENCE_LENGTH, BATCH_SIZE, shuffle=False, device=device)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # Initialize model, loss function, and optimizer
    model = get_model(MODEL_TYPE, len(stoi)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Training loop
    best_val = float("inf")
    train_hist, val_hist = [], []
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        # Training epoch
        tr_loss = run_epoch(model, train_loader, criterion, optimizer, device, 
                           clip=GRADIENT_CLIP, train=True)
        
        # Validation epoch
        va_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        
        # Record losses
        train_hist.append(tr_loss)
        val_hist.append(va_loss)
        
        print(f"Epoch {epoch}/{EPOCHS} | train CE {tr_loss:.4f} | val CE {va_loss:.4f}")

        # Save best model
        if va_loss < best_val:
            best_val = va_loss
            ckpt = os.path.join(OUTPUT_DIR, f"best_{MODEL_TYPE}.pt")
            
            # Create args dict for compatibility with eval/generate scripts
            args_dict = {
                "data_path": DATA_PATH,
                "model": MODEL_TYPE,
                "seq_len": SEQUENCE_LENGTH,
                "batch_size": BATCH_SIZE,
                "embed_size": EMBED_SIZE,
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
                "n_heads": N_HEADS,
                "ff_size": FF_SIZE,
                "epochs": EPOCHS,
                "lr": LEARNING_RATE,
                "seed": SEED,
                "clip": GRADIENT_CLIP,
                "out_dir": OUTPUT_DIR
            }
            
            save_ckpt(ckpt, model, stoi, itos, args_dict)
            print(f"Saved best model to {ckpt}")

    total_time = time.time() - t0
    print(f"Training completed in {total_time:.1f}s")
    print(f"Best validation loss: {best_val:.4f}")

    # Save training curves
    png_path = os.path.join(OUTPUT_DIR, f"loss_{MODEL_TYPE}.png")
    title = f"{MODEL_TYPE} — seq_len={SEQUENCE_LENGTH}, batch_size={BATCH_SIZE}, dropout={DROPOUT}"
    plot_curves(train_hist, val_hist, png_path, title)
    print(f"Loss curves saved to {png_path}")
    
    # Save metrics
    metrics_path = os.path.join(OUTPUT_DIR, f"metrics_{MODEL_TYPE}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        import json
        json.dump({
            "train_ce": train_hist, 
            "val_ce": val_hist, 
            "best_val_ce": best_val, 
            "train_time_s": total_time,
            "config": args_dict
        }, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()
