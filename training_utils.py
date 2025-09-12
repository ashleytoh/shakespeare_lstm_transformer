#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training and Evaluation Utilities for Shakespeare Language Model
Handles model training, evaluation, and text generation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import os
import time
import math
import matplotlib
matplotlib.use("Agg")  # headless plots
import matplotlib.pyplot as plt


def train_model(model, train_loader, val_loader, device, model_name="model", 
                epochs=5, learning_rate=3e-4, gradient_clip=1.0, output_dir="reports"):
    """
    Training loop with mixed precision and efficiency improvements
    
    Args:
        model: PyTorch model to train
        train_loader, val_loader: Data loaders
        device: PyTorch device
        model_name: Name for saving files
        epochs: Number of training epochs
        learning_rate: Learning rate
        gradient_clip: Gradient clipping value
        output_dir: Directory to save results
    
    Returns:
        tuple: (train_losses, val_losses, training_time)
    """
    print("\n" + "=" * 60)
    print(f"TRAINING {model_name.upper()}")
    print("=" * 60)
    
    # Record training start time
    training_start_time = time.time()
    
    criterion = nn.CrossEntropyLoss()
    # AdamW with weight decay for better generalization
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Mixed precision training for 2x speedup
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # Learning rate scheduler with warmup
    total_steps = len(train_loader) * epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate, total_steps=total_steps,
        pct_start=0.1, anneal_strategy='cos'
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_name.upper()}")
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
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        train_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", 
                           miniters=len(train_loader)//20)  # Update every 5%
        
        for batch_idx, (x_batch, y_batch) in enumerate(progress_bar):
            x_batch, y_batch = x_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast() if scaler else torch.enable_grad():
                # Forward pass - handle different model types
                if hasattr(model, 'lstm'):  # LSTM model
                    output, _ = model(x_batch, None)  # Don't carry hidden states across batches
                else:  # Transformer model
                    output = model(x_batch)
                
                # Calculate loss
                loss = criterion(output.view(-1, output.size(-1)), y_batch.view(-1))
            
            # Mixed precision backward pass
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
            
            total_train_loss += loss.item()
            train_batches += 1
            
            # Update learning rate after each batch
            scheduler.step()
            
            # Update progress bar
            if batch_idx % max(1, len(train_loader)//20) == 0:
                progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        # Validation
        model.eval()
        total_val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
                
                with torch.cuda.amp.autocast() if scaler else torch.enable_grad():
                    if hasattr(model, 'lstm'):  # LSTM model
                        output, _ = model(x_batch, None)
                    else:  # Transformer model
                        output = model(x_batch)
                    
                    loss = criterion(output.view(-1, output.size(-1)), y_batch.view(-1))
                
                total_val_loss += loss.item()
                val_batches += 1
        
        # Calculate average losses
        avg_train_loss = total_train_loss / train_batches
        avg_val_loss = total_val_loss / val_batches
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Early stopping and save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            os.makedirs(output_dir, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss,
                'epoch': epoch + 1
            }, os.path.join(output_dir, f'best_{model_name}_model.pt'))
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


def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test set
    
    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        device: PyTorch device
    
    Returns:
        tuple: (test_loss, test_perplexity)
    """
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
            
            if hasattr(model, 'lstm'):  # LSTM model
                output, _ = model(x_batch, None)
            else:  # Transformer model
                output = model(x_batch)
            
            loss = criterion(output.view(-1, output.size(-1)), y_batch.view(-1))
            total_loss += loss.item()
            total_batches += 1
    
    avg_loss = total_loss / total_batches
    perplexity = math.exp(avg_loss)
    
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Perplexity: {perplexity:.2f}")
    
    return avg_loss, perplexity


def generate_text(model, start_text, char_to_idx, idx_to_char, max_len=200, 
                 temperature=1.0, device='cpu', sequence_length=128):
    """
    Generate text using the trained model
    
    Args:
        model: Trained PyTorch model
        start_text: Starting prompt text
        char_to_idx, idx_to_char: Character mappings
        max_len: Maximum length to generate
        temperature: Sampling temperature
        device: PyTorch device
        sequence_length: Model sequence length
    
    Returns:
        str: Generated text
    """
    model.eval()
    
    # Encode start text
    input_ids = torch.tensor([char_to_idx.get(char, 0) for char in start_text], 
                            dtype=torch.long).unsqueeze(0).to(device)
    generated_text = list(start_text)
    hidden = None
    
    with torch.no_grad():
        if hasattr(model, 'lstm'):  # LSTM model
            for _ in range(max_len):
                output, hidden = model(input_ids[:, -1:], hidden)
                output = output[:, -1, :] / temperature
                probabilities = F.softmax(output, dim=-1)
                
                next_char_id = torch.multinomial(probabilities, num_samples=1).item()
                next_char = idx_to_char[next_char_id]
                generated_text.append(next_char)
                
                input_ids = torch.tensor([[next_char_id]], dtype=torch.long).to(device)
        else:  # Transformer model
            current_seq = input_ids
            for _ in range(max_len):
                if current_seq.size(1) >= sequence_length:
                    current_seq = current_seq[:, -sequence_length//2:]  # Truncate context
                
                output = model(current_seq)
                output = output[:, -1, :] / temperature
                probabilities = F.softmax(output, dim=-1)
                
                next_char_id = torch.multinomial(probabilities, num_samples=1)
                next_char = idx_to_char[next_char_id.item()]
                generated_text.append(next_char)
                
                current_seq = torch.cat([current_seq, next_char_id], dim=1)
    
    return ''.join(generated_text)


def plot_training_curves(train_losses, val_losses, model_name="model", output_dir="reports"):
    """
    Plot and save training curves
    
    Args:
        train_losses, val_losses: Lists of losses per epoch
        model_name: Name for the plot
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.title(f'{model_name.upper()} Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{model_name}_training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {output_dir}/{model_name}_training_curves.png")
