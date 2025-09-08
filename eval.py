#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Evaluation Script
Evaluates trained models on validation or test sets and computes perplexity
"""

import torch
import math
from torch.utils.data import DataLoader
from utils import load_ckpt, encode, split_indices, make_loader, load_text
from models.lstm import LSTMLanguageModel
from models.transformer import TinyTransformerLM
import torch.nn as nn

# =============================================================================
# EVALUATION CONFIGURATION - Configure your evaluation here
# =============================================================================

# Model Configuration
CHECKPOINT_PATH = "reports/best_lstm.pt"        # Path to trained model checkpoint
EVALUATION_SPLIT = "test"                       # "val" or "test" split to evaluate on

# =============================================================================

def build_from_args(args_saved, vocab_size):
    """Rebuild model from saved configuration"""
    if args_saved["model"] == "lstm":
        return LSTMLanguageModel(
            vocab_size=vocab_size,
            embed_size=args_saved["embed_size"],
            hidden_size=args_saved["hidden_size"],
            num_layers=args_saved["num_layers"],
            dropout=args_saved["dropout"]
        )
    else:
        return TinyTransformerLM(
            vocab_size=vocab_size,
            d_model=args_saved["embed_size"],
            n_layers=args_saved["num_layers"],
            n_heads=args_saved["n_heads"],
            d_ff=args_saved["ff_size"],
            dropout=args_saved["dropout"],
            max_len=args_saved["seq_len"]
        )

def main():
    """Main evaluation function"""
    print("Model Evaluation")
    print("=" * 40)
    
    # Load checkpoint
    print(f"Loading model from {CHECKPOINT_PATH}...")
    ckpt = load_ckpt(CHECKPOINT_PATH, map_location="cpu")
    stoi, itos, saved = ckpt["stoi"], ckpt["itos"], ckpt["args"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Rebuild and load model
    model = build_from_args(saved, vocab_size=len(stoi)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Loaded {saved['model']} model with {len(stoi)} vocabulary size")

    # Recreate the dataset
    print(f"Loading dataset from {saved['data_path']}...")
    text = load_text(saved["data_path"])
    ids = encode(text, stoi)
    
    # Get the appropriate split
    (s0, e0), (s1, e1), (s2, e2) = split_indices(len(ids))
    if EVALUATION_SPLIT == "val":
        split_ids = ids[s1:e1]
        split_name = "validation"
    else:
        split_ids = ids[s2:e2]
        split_name = "test"
    
    print(f"Evaluating on {split_name} split ({len(split_ids)} tokens)")
    
    # Create data loader
    loader = make_loader(split_ids, saved["seq_len"], saved["batch_size"], 
                        shuffle=False, device=device)

    # Evaluate model
    criterion = nn.CrossEntropyLoss()
    total_loss, n_batches = 0.0, 0
    
    print("Computing loss...")
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            
            # Forward pass - handle different model types
            if hasattr(model, "lstm"):
                logits, _ = model(x, None)
            else:
                logits = model(x)
                
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += float(loss.detach().cpu())
            n_batches += 1

    # Calculate metrics
    ce = total_loss / max(n_batches, 1)
    ppl = math.exp(ce)
    
    # Display results
    print("\nEvaluation Results:")
    print(f"{EVALUATION_SPLIT.upper()} Cross-Entropy: {ce:.4f}")
    print(f"{EVALUATION_SPLIT.upper()} Perplexity: {ppl:.2f}")

if __name__ == "__main__":
    main()
