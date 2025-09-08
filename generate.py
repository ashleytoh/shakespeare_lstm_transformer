#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shakespeare Text Generation
Generates text using trained LSTM or Transformer models with temperature sampling
"""

import torch
import torch.nn.functional as F
from utils import load_ckpt
from models.lstm import LSTMLanguageModel
from models.transformer import TinyTransformerLM

# =============================================================================
# GENERATION CONFIGURATION - Configure your text generation here
# =============================================================================

# Model Configuration
CHECKPOINT_PATH = "reports/best_lstm.pt"        # Path to trained model checkpoint
PROMPT = ""                                     # Starting prompt (empty for random start)
TEMPERATURES = [0.7, 1.0, 1.3]                # Temperature values for sampling diversity
MAX_NEW_TOKENS = 300                           # Number of tokens to generate

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

def sample(model, start_ids, max_new_tokens, temperature, device):
    """Generate text using temperature-controlled sampling"""
    model.eval()
    x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)
    generated = start_ids[:]
    hidden = None
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass - handle different model types
            if hasattr(model, "lstm"):
                logits, hidden = model(x[:, -1:], hidden)  # feed last token for RNN
            else:
                logits = model(x)[:, -1:, :]
                
            # Apply temperature and sample
            logits = logits[:, -1, :] / max(1e-6, temperature)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            
            # Update generated sequence
            generated.append(next_id)
            x = torch.tensor([generated], dtype=torch.long, device=device)
            
    return generated

def main():
    """Main text generation function"""
    print("Shakespeare Text Generator")
    print("=" * 50)
    
    # Load checkpoint
    print(f"Loading model from {CHECKPOINT_PATH}...")
    ckpt = load_ckpt(CHECKPOINT_PATH, map_location="cpu")
    stoi, itos, saved = ckpt["stoi"], ckpt["itos"], ckpt["args"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Rebuild and load model
    model = build_from_args(saved, vocab_size=len(stoi)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"Loaded {saved['model']} model with {len(stoi)} vocabulary size")

    # Encode prompt (empty prompt is allowed)
    start_ids = [stoi.get(ch, 0) for ch in PROMPT]
    prompt_text = PROMPT if PROMPT else "[Random start]"
    print(f"Starting prompt: '{prompt_text}'")
    print(f"Generating {MAX_NEW_TOKENS} tokens with temperatures: {TEMPERATURES}")
    print()

    # Generate text at different temperatures
    for T in TEMPERATURES:
        print(f"=== Temperature {T} ===")
        out_ids = sample(model, start_ids=start_ids, max_new_tokens=MAX_NEW_TOKENS,
                         temperature=T, device=device)
        text = "".join(itos[i] for i in out_ids)
        print(text)
        print()

if __name__ == "__main__":
    main()
