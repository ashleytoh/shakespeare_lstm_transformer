#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration settings
All hyperparameters and experiment configurations
"""

# =============================================================================
# CORE CONFIGURATION
# =============================================================================

# Data Configuration
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

# =============================================================================
# ABLATION STUDIES CONFIGURATION
# =============================================================================

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
# OPTIMIZATION SETTINGS
# =============================================================================

# PyTorch optimization settings
import torch
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
torch.backends.cudnn.benchmark = True
