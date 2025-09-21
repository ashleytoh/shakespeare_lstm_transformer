# Shakespeare Text Generation: LSTM vs. Transformer Language Models

This repository implements and compares LSTM and Transformer language models for character-level text generation on Shakespeare's corpus.

## Overview

The project trains neural language models to generate Shakespeare-like text using:
- **LSTM**: Recurrent architecture with memory cells
- **Transformer**: Attention-based architecture with self-attention

## Key Features

- **Modular Architecture**: Clean separation of concerns across multiple modules
- **Character-level tokenization** for detailed text modeling
- **Temperature-controlled text generation** (0.7, 1.0, 1.3)
- **Evaluation** (cross-entropy loss, perplexity)
- **Ablation studies** comparing different hyperparameters
- **Training visualization** and model checkpointing
- **Mixed precision training** for efficiency
- **Early stopping** to prevent overfitting

## Repository Structure

```
├── main.py                     # Main orchestration script
├── config.py                   # Configuration and hyperparameters
├── data_utils.py              # Data loading and preprocessing
├── training_utils.py          # Training and evaluation functions
├── evaluation_utils.py        # Experiment management and ablation studies
├── models/                    # Model implementations
│   ├── __init__.py
│   ├── lstm_model.py         # LSTM language model
│   ├── transformer_model.py  # Transformer language model
│   └── model_factory.py      # Model creation factory
├── data/
│   └── input.txt             # Shakespeare text corpus
├── reports_lstm/             # LSTM experiment results
│   ├── *.png                # Training curves
│   ├── *.txt                # Generated text samples
│   ├── *.pt                 # Model checkpoints
│   └── comprehensive_results.json
└── reports_transformer/      # Transformer experiment results
    ├── *.png                # Training curves
    ├── *.txt                # Generated text samples
    ├── *.pt                 # Model checkpoints
    └── comprehensive_results.json
```

## Module Descriptions

### Core Modules

- **`main.py`**: Entry point that orchestrates the entire pipeline
- **`config.py`**: Centralized configuration with all hyperparameters
- **`data_utils.py`**: Data loading, preprocessing, and dataset creation
- **`training_utils.py`**: Model training, evaluation, and text generation
- **`evaluation_utils.py`**: Experiment management and ablation studies

### Model Modules

- **`models/lstm_model.py`**: LSTM language model implementation
- **`models/transformer_model.py`**: Transformer model with attention blocks
- **`models/model_factory.py`**: Factory function for creating models

## Usage

### Quick Start

Run the complete pipeline with default settings:

```bash
python main.py
```

### Configuration

Edit `config.py` to modify:

```python
# Model selection
MODEL_TYPE = "transformer"  # or "lstm"

# Training parameters
EPOCHS = 5
LEARNING_RATE = 3e-4
BATCH_SIZE = 128
SEQUENCE_LENGTH = 128

# Architecture parameters
EMBED_SIZE = 128
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.1
```

### Ablation Studies

The system automatically runs ablation studies comparing:

1. **Dropout Study**: 0.0 vs 0.2 dropout rates
2. **Context Length Study**: 128 vs 256 sequence lengths

Configure studies in `config.py`:

```python
ABLATION_STUDIES = [
    {
        'name': 'dropout_study',
        'description': 'Dropout: 0.0 vs 0.2',
        'configs': [
            {'dropout': 0.0, 'label': 'dropout_0.0'},
            {'dropout': 0.2, 'label': 'dropout_0.2'}
        ]
    }
]
```

## Model Architectures

### LSTM Model
- Character embedding layer with dropout
- Multi-layer LSTM with configurable hidden size
- Output dropout for regularization
- Linear projection to vocabulary size

### Transformer Model
- Token and positional embeddings
- Multi-head self-attention blocks
- Feed-forward networks with GELU activation
- Layer normalization and residual connections
- Causal masking for autoregressive generation

## Optimization Features

- **Mixed Precision Training**: 2x speedup on compatible GPUs
- **Learning Rate Scheduling**: OneCycleLR with warmup
- **Gradient Clipping**: Prevents exploding gradients
- **Early Stopping**: Prevents overfitting
- **Model Compilation**: PyTorch 2.0 optimization when available
- **Efficient Data Loading**: Multi-worker dataloaders with prefetching

## Output Files

For each experiment, the system generates:

- **Training curves**: `{experiment}_training_curves.png`
- **Text samples**: `{experiment}_generation_T{temperature}.txt`
- **Model checkpoints**: `best_{experiment}_model.pt`
- **Comprehensive results**: `comprehensive_results.json`

## Requirements

```bash
pip install torch>=1.12.0
pip install matplotlib>=3.5.0
pip install tqdm>=4.62.0
pip install numpy>=1.21.0
```
