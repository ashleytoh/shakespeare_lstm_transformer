#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Processing Utilities for Shakespeare Language Model
Handles text loading, vocabulary building, and dataset creation
"""

import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_shakespeare_text(data_path, reduction_factor=1):
    """
    Load and preview the Shakespeare text corpus
    
    Args:
        data_path: Path to the text file
        reduction_factor: Fraction of data to use (for faster training)
    
    Returns:
        str: Processed text
    """
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Store original text stats for reporting
    original_length = len(text)
    
    # Use only a fraction of the data for faster training
    if reduction_factor < 1.0:
        reduced_length = int(len(text) * reduction_factor)
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
    """
    Build character-level vocabulary
    
    Args:
        text: Input text string
    
    Returns:
        tuple: (char_to_idx, idx_to_char, vocab_size)
    """
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


def decode_text(indices, idx_to_char):
    """Convert numerical indices back to text"""
    return ''.join([idx_to_char[idx] for idx in indices])


class ShakespeareDataset(Dataset):
    """Character-level Shakespeare dataset for sequence modeling"""
    
    def __init__(self, data, seq_len):
        """
        Initialize dataset
        
        Args:
            data: List of character indices
            seq_len: Sequence length for training
        """
        # Pre-convert to tensor for faster access
        self.data = torch.tensor(data, dtype=torch.long)
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        # Direct tensor slicing for efficiency
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]
        return x, y


def create_data_loaders(train_data, val_data, test_data, seq_len, batch_size, device):
    """
    Create optimized data loaders for training, validation, and testing
    
    Args:
        train_data, val_data, test_data: Lists of character indices
        seq_len: Sequence length
        batch_size: Batch size
        device: PyTorch device
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = ShakespeareDataset(train_data, seq_len)
    val_dataset = ShakespeareDataset(val_data, seq_len)
    test_dataset = ShakespeareDataset(test_data, seq_len)
    
    # Optimize data loader settings
    num_workers = 4 if device.type == 'cuda' else 2
    pin_memory = device.type == 'cuda'
    
    # Optimize batch size for GPU memory
    if device.type == 'cuda':
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            if gpu_memory > 8:  # High-memory GPU
                batch_size = min(batch_size * 2, 256)  # Up to 256
        except:
            pass
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=pin_memory, 
        persistent_workers=num_workers > 0, prefetch_factor=4
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0, prefetch_factor=2
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0, prefetch_factor=2
    )
    
    return train_loader, val_loader, test_loader


def split_data(data, train_split=0.8, val_split=0.1):
    """
    Split data into train/validation/test sets
    
    Args:
        data: List of character indices
        train_split: Fraction for training
        val_split: Fraction for validation
    
    Returns:
        tuple: (train_data, val_data, test_data)
    """
    n_train = int(len(data) * train_split)
    n_val = int(len(data) * val_split)
    
    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]
    
    return train_data, val_data, test_data
