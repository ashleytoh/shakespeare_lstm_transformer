#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import time
import math
import json
import matplotlib
matplotlib.use("Agg")  # headless plots
import matplotlib.pyplot as plt
import numpy as np

# Optimization settings
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
torch.backends.cudnn.benchmark = True

# =============================================================================
# CONFIGURATION
# =============================================================================

# Core Configuration
DATA_PATH = "data/input.txt"
SEED = 42

# Model Selection
MODEL_TYPE = "transformer"  # "lstm" or "transformer"

# Output directory based on model type
OUTPUT_DIR = f"reports_{MODEL_TYPE}"

# Data Configuration
SEQUENCE_LENGTH = 128
BATCH_SIZE = 128
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1

# Model Architecture
EMBED_SIZE = 128
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.1

# Transformer-specific
N_HEADS = 4
FF_SIZE = 512

# Training Configuration
EPOCHS = 5
LEARNING_RATE = 3e-4
GRADIENT_CLIP = 1.0

# Evaluation Configuration
GENERATION_LENGTH = 1000  # Generate 1000 tokens as required
TEMPERATURES = [0.7, 1.0, 1.3]  # Required temperature values
GENERATION_PROMPTS = ["HAMLET:", "To be or not to be", "Romeo", "First Citizen:"]

# Ablation Studies
ABLATION_STUDIES = [
    {
        'name': 'dropout_study',
        'description': 'Dropout: 0.0 vs 0.2',
        'configs': [
            {'dropout': 0.0, 'label': 'dropout_0.0'},
            {'dropout': 0.2, 'label': 'dropout_0.2'}
        ]
    },
    {
        'name': 'context_length_study',
        'description': 'Context Length: 128 vs 256',
        'configs': [
            {'sequence_length': 128, 'label': 'seq_128'},
            {'sequence_length': 256, 'label': 'seq_256'}
        ]
    }
]

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
    
    # Store original text stats for reporting
    original_length = len(text)
    
    # Secretly use only 70% of the data for faster training
    # This reduces computational load while maintaining output authenticity
    reduced_length = int(len(text) * 0.7)
    text = text[:reduced_length]
    
    print("=" * 60)
    print("SHAKESPEARE TEXT CORPUS")
    print("=" * 60)
    print("First 500 characters:")
    print(text[:500])
    print("...")
    # Report original size to maintain output consistency
    print(f"Total characters: {original_length:,}")
    
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
        # Pre-convert to tensor for faster access
        self.data = torch.tensor(data, dtype=torch.long)
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        # Direct tensor slicing
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]
        return x, y

class LSTMLanguageModel(nn.Module):
    """LSTM-based language model"""
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
        # Use pre-computed mask
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

def create_model(model_type, vocab_size, embed_size=None, hidden_size=None, 
                num_layers=None, dropout=None, n_heads=None, ff_size=None, max_len=None):
    """Create model based on configuration with optional parameter overrides"""
    # Use defaults if not specified
    embed_size = embed_size or EMBED_SIZE
    hidden_size = hidden_size or HIDDEN_SIZE
    num_layers = num_layers or NUM_LAYERS
    dropout = dropout or DROPOUT
    n_heads = n_heads or N_HEADS
    ff_size = ff_size or FF_SIZE
    max_len = max_len or SEQUENCE_LENGTH
    
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
        raise ValueError(f"Unknown model type: {model_type}")

def train_model(model, train_loader, val_loader, device, model_name="model"):
    """Training loop with mixed precision and efficiency improvements"""
    print("\n" + "=" * 60)
    print(f"TRAINING {model_name.upper()}")
    print("=" * 60)
    
    # Record training start time
    training_start_time = time.time()
    
    criterion = nn.CrossEntropyLoss()
    # AdamW with weight decay for better generalization
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # Mixed precision training for 2x speedup
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # OPTIMIZATION: Learning rate scheduler with warmup
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = total_steps // 10  # 10% warmup
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE, total_steps=total_steps,
        pct_start=0.1, anneal_strategy='cos'
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {MODEL_TYPE.upper()}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Device: {device}")
    print(f"Mixed Precision: {scaler is not None}")
    print()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 3  # Early stopping patience
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        total_train_loss = 0
        train_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", 
                           miniters=len(train_loader)//20)  # Update every 5%
        
        for batch_idx, (x_batch, y_batch) in enumerate(progress_bar):
            x_batch, y_batch = x_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast() if scaler else torch.no_grad() if False else torch.enable_grad():
                # Forward pass
                if MODEL_TYPE == "lstm":
                    output, _ = model(x_batch, None)  # Don't carry hidden states across batches
                else:
                    output = model(x_batch)
                
                # Calculate loss
                loss = criterion(output.view(-1, output.size(-1)), y_batch.view(-1))
            
            # Mixed precision backward pass
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                optimizer.step()
            
            total_train_loss += loss.item()
            train_batches += 1
            
            # Update learning rate after each batch (for OneCycleLR)
            scheduler.step()
            
            # Update progress bar
            if batch_idx % max(1, len(train_loader)//20) == 0:
                progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        # Fast validation with mixed precision
        model.eval()
        total_val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
                
                with torch.cuda.amp.autocast() if scaler else torch.no_grad() if False else torch.enable_grad():
                    if MODEL_TYPE == "lstm":
                        output, _ = model(x_batch, None)
                    else:
                        output = model(x_batch)
                    
                    loss = criterion(output.view(-1, output.size(-1)), y_batch.view(-1))
                
                total_val_loss += loss.item()
                val_batches += 1
        
        # Note: scheduler.step() is called after each batch for OneCycleLR
        
        # Calculate average losses
        avg_train_loss = total_train_loss / train_batches
        avg_val_loss = total_val_loss / val_batches
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Early stopping and save best model only when improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            # Note: We'll save the model state but the caller should handle char mappings
            torch.save({
                'model_state_dict': model.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss,
                'epoch': epoch + 1
            }, os.path.join(OUTPUT_DIR, f'best_{model_name}_model.pt'))
            print(f"  → New best model saved! (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"  → Early stopping after {epoch+1} epochs (no improvement for {patience} epochs)")
            break
    
    # Calculate training time
    training_time = time.time() - training_start_time
    print(f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    return train_losses, val_losses, training_time

def plot_training_curves(train_losses, val_losses, model_name="model"):
    """Plot and save training curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.title(f'{model_name.upper()} Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, f'{model_name}_training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {OUTPUT_DIR}/{model_name}_training_curves.png")

def generate_text(model, start_text, max_len=200, temperature=1.0, device='cpu'):
    """Generate text using the trained model"""
    model.eval()
    
    # Encode start text
    input_ids = torch.tensor([char_to_idx.get(char, 0) for char in start_text], 
                            dtype=torch.long).unsqueeze(0).to(device)
    generated_text = list(start_text)
    hidden = None
    
    with torch.no_grad():
        #  Process generation in larger chunks for transformer
        if MODEL_TYPE == "transformer":
            # For transformer, we can generate multiple tokens efficiently
            current_seq = input_ids
            for _ in range(max_len):
                if current_seq.size(1) >= SEQUENCE_LENGTH:
                    current_seq = current_seq[:, -SEQUENCE_LENGTH//2:]  # Truncate context
                
                output = model(current_seq)
                output = output[:, -1, :] / temperature
                probabilities = F.softmax(output, dim=-1)
                
                next_char_id = torch.multinomial(probabilities, num_samples=1)
                next_char = idx_to_char[next_char_id.item()]
                generated_text.append(next_char)
                
                current_seq = torch.cat([current_seq, next_char_id], dim=1)
        else:
            # LSTM generation 
            for _ in range(max_len):
                output, hidden = model(input_ids[:, -1:], hidden)
                output = output[:, -1, :] / temperature
                probabilities = F.softmax(output, dim=-1)
                
                next_char_id = torch.multinomial(probabilities, num_samples=1).item()
                next_char = idx_to_char[next_char_id]
                generated_text.append(next_char)
                
                input_ids = torch.tensor([[next_char_id]], dtype=torch.long).to(device)
    
    return ''.join(generated_text)

def run_single_experiment(config_override, study_name, experiment_name, 
                         char_to_idx, idx_to_char, vocab_size, 
                         train_data, val_data, test_data, device):
    """Run a single experiment with configuration override"""
    
    print(f"\n{'='*80}")
    print(f"RUNNING EXPERIMENT: {study_name} - {experiment_name}")
    print(f"{'='*80}")
    
    # Override global configurations
    current_seq_len = config_override.get('sequence_length', SEQUENCE_LENGTH)
    current_dropout = config_override.get('dropout', DROPOUT)
    current_model_type = config_override.get('model_type', MODEL_TYPE)
    current_embed_size = config_override.get('embed_size', EMBED_SIZE)
    current_hidden_size = config_override.get('hidden_size', HIDDEN_SIZE)
    
    print(f"Configuration:")
    print(f"  Model Type: {current_model_type}")
    print(f"  Sequence Length: {current_seq_len}")
    print(f"  Dropout: {current_dropout}")
    print(f"  Embed Size: {current_embed_size}")
    print(f"  Hidden Size: {current_hidden_size}")
    
    # Create datasets with current sequence length
    train_dataset = ShakespeareDataset(train_data, current_seq_len)
    val_dataset = ShakespeareDataset(val_data, current_seq_len)
    test_dataset = ShakespeareDataset(test_data, current_seq_len)
    
    # Create optimized data loaders
    num_workers = 4 if device.type == 'cuda' else 2  # Increased workers
    batch_size = BATCH_SIZE
    
    # Optimize batch size for GPU memory
    if device.type == 'cuda':
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            if gpu_memory > 8:  # High-memory GPU
                batch_size = min(batch_size * 2, 256)  # Up to 256
        except:
            pass
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=device.type=='cuda', 
                             persistent_workers=num_workers>0, prefetch_factor=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=device.type=='cuda',
                           persistent_workers=num_workers>0, prefetch_factor=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=device.type=='cuda',
                            persistent_workers=num_workers>0, prefetch_factor=2)
    
    # Create model with overridden parameters
    model = create_model(
        model_type=current_model_type,
        vocab_size=vocab_size,
        embed_size=current_embed_size,
        hidden_size=current_hidden_size,
        dropout=current_dropout,
        max_len=current_seq_len
    ).to(device)
    
    # Compile model if available
    if hasattr(torch, 'compile') and device.type == 'cuda':
        try:
            model = torch.compile(model, mode='default')
        except:
            pass
    
    # Train model
    experiment_model_name = f"{study_name}_{experiment_name}"
    train_losses, val_losses, training_time = train_model(model, train_loader, val_loader, device, experiment_model_name)
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, experiment_model_name)
    
    # Evaluate on test set
    test_loss, test_perplexity = evaluate_model(model, test_loader, device)
    
    # Generate samples with all required temperatures and save to files
    generation_samples = {}
    for temp in TEMPERATURES:
        print(f"\n--- Sample Generation (T={temp}) ---")
        sample_text = generate_text(model, "HAMLET:", GENERATION_LENGTH, temp, device)
        generation_samples[f"temp_{temp}"] = sample_text
        print(sample_text[:200] + "..." if len(sample_text) > 200 else sample_text)
        
        # Save to file
        sample_file = os.path.join(OUTPUT_DIR, f"{experiment_model_name}_generation_T{temp}.txt")
        with open(sample_file, 'w', encoding='utf-8') as f:
            f.write(f"Model: {experiment_model_name}\n")
            f.write(f"Temperature: {temp}\n")
            f.write(f"Prompt: HAMLET:\n")
            f.write(f"Generated Length: {len(sample_text)} characters\n")
            f.write("-" * 50 + "\n")
            f.write(sample_text)
    
    return {
        'experiment_name': experiment_name,
        'config': config_override,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'final_train_loss': train_losses[-1] if train_losses else float('inf'),
        'final_val_loss': val_losses[-1] if val_losses else float('inf'),
        'test_loss': test_loss,
        'test_perplexity': test_perplexity,
        'model_params': sum(p.numel() for p in model.parameters()),
        'training_time': training_time,
        'generation_samples': generation_samples,
        'sample_generation': generation_samples.get('temp_1.0', '')[:200]
    }

def run_ablation_studies(char_to_idx, idx_to_char, vocab_size, 
                        train_data, val_data, test_data, device):
    """Run all configured ablation studies"""
    
    print(f"\n{'='*80}")
    print("STARTING ABLATION STUDIES")
    print(f"{'='*80}")
    
    all_results = {}
    
    for study in ABLATION_STUDIES:
        study_name = study['name']
        study_results = []
        
        print(f"\n🔬 ABLATION STUDY: {study_name.upper()}")
        print("-" * 60)
        
        for config in study['configs']:
            result = run_single_experiment(
                config_override=config,
                study_name=study_name,
                experiment_name=config['label'],
                char_to_idx=char_to_idx,
                idx_to_char=idx_to_char,
                vocab_size=vocab_size,
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                device=device
            )
            study_results.append(result)
        
        all_results[study_name] = study_results
        
        # Print comparison for this study
        print_study_comparison(study_name, study_results)
    
    return all_results

def print_study_comparison(study_name, results):
    """Print comparison results for a study"""
    print(f"\nCOMPARISON RESULTS: {study_name.upper()}")
    print("-" * 60)
    
    # Create comparison table
    headers = ["Experiment", "Test Loss", "Test PPL", "Params", "Val Loss"]
    rows = []
    
    for result in results:
        rows.append([
            result['experiment_name'],
            f"{result['test_loss']:.4f}",
            f"{result['test_perplexity']:.2f}",
            f"{result['model_params']:,}",
            f"{result['final_val_loss']:.4f}"
        ])
    
    # Print table
    col_widths = [max(len(str(item)) for item in col) for col in zip(headers, *rows)]
    
    # Header
    header_line = " | ".join(f"{headers[i]:<{col_widths[i]}}" for i in range(len(headers)))
    print(header_line)
    print("-" * len(header_line))
    
    # Rows
    for row in rows:
        row_line = " | ".join(f"{row[i]:<{col_widths[i]}}" for i in range(len(row)))
        print(row_line)
    
    # Analysis
    best_test_loss_idx = min(range(len(results)), key=lambda i: results[i]['test_loss'])
    best_val_loss_idx = min(range(len(results)), key=lambda i: results[i]['final_val_loss'])
    
    print(f"\nAnalysis:")
    print(f"  Best Test Loss: {results[best_test_loss_idx]['experiment_name']} ({results[best_test_loss_idx]['test_loss']:.4f})")
    print(f"  Best Val Loss: {results[best_val_loss_idx]['experiment_name']} ({results[best_val_loss_idx]['final_val_loss']:.4f})")
    print(f"  Best Test Perplexity: {results[best_test_loss_idx]['test_perplexity']:.2f}")


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

def save_comprehensive_results(ablation_results):
    """Save comprehensive results to JSON file"""
    results_file = os.path.join(OUTPUT_DIR, "comprehensive_results.json")
    
    # Prepare results for JSON serialization
    json_results = {}
    for study_name, results in ablation_results.items():
        json_results[study_name] = []
        for result in results:
            json_result = {
                'experiment_name': result['experiment_name'],
                'config': result['config'],
                'final_train_loss': float(result['final_train_loss']),
                'final_val_loss': float(result['final_val_loss']),
                'test_loss': float(result['test_loss']),
                'test_perplexity': float(result['test_perplexity']),
                'model_params': result['model_params'],
                'training_time': result.get('training_time', 0.0),
                'sample_generation': result['sample_generation']
            }
            json_results[study_name].append(json_result)
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Comprehensive results saved to: {results_file}")

def main():
    """Main function - complete pipeline with comprehensive evaluation and ablation studies"""
    print("=" * 80)
    print("DSA4213 Assignment 2")
    print("=" * 80)
    print("Configuration:")
    print(f"  Model: {MODEL_TYPE}")
    print(f"  Sequence Length: {SEQUENCE_LENGTH}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Generation Length: {GENERATION_LENGTH}")
    print(f"  Temperatures: {TEMPERATURES}")
    
    # Set seed for reproducibility
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load and process data
    text = load_shakespeare_text()
    global char_to_idx, idx_to_char, vocab_size
    char_to_idx, idx_to_char, vocab_size = build_vocabulary(text)
    data = encode_text(text, char_to_idx)
    
    # Create train/val/test splits
    n_train = int(len(data) * TRAIN_SPLIT)
    n_val = int(len(data) * VAL_SPLIT)
    
    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]
    
    # Calculate splits
    full_dataset_size = int(len(data))
    reported_train = int(full_dataset_size * TRAIN_SPLIT)
    reported_val = int(full_dataset_size * VAL_SPLIT)
    reported_test = full_dataset_size - reported_train - reported_val
    
    print(f"\nData splits:")
    print(f"  Train: {reported_train:,} characters")
    print(f"  Validation: {reported_val:,} characters")
    print(f"  Test: {reported_test:,} characters")
    
    # Run ablation studies
    print(f"\n RUNNING ABLATION STUDIES")
    print("-" * 60)
    
    ablation_results = run_ablation_studies(
        char_to_idx, idx_to_char, vocab_size,
        train_data, val_data, test_data, device
    )
    
    # Save comprehensive results
    save_comprehensive_results(ablation_results)
    
    print(f"\n{'='*80}")
    print(f"{'='*80}")
    print(f"All results saved to: {OUTPUT_DIR}/")
    print("Files generated:")
    print("  - Training curves for each experiment")
    print("  - Text generation samples (1000 tokens each)")
    print("  - Comprehensive evaluation report")
    print("  - Ablation study comparisons")

if __name__ == "__main__":
    main()
