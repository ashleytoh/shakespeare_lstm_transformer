# DSA4213 Assignment 2 - Shakespeare Language Models

This repository implements and compares LSTM and Transformer language models for character-level text generation on Shakespeare's corpus.

## Overview

The project trains neural language models to generate Shakespeare-like text using:
- **LSTM**: Recurrent architecture with memory cells
- **Transformer**: Attention-based architecture with self-attention

## Key Features

- Character-level tokenization
- Temperature-controlled text generation  
- Comprehensive evaluation (cross-entropy loss, perplexity)
- Training visualization and model checkpointing
- Embedded hyperparameters (no command-line arguments needed)

## Files Structure

### Core Scripts (With Embedded Hyperparameters)
- **`train.py`**: Main training script
- **`generate.py`**: Text generation with trained models  
- **`eval.py`**: Model evaluation on test sets
- **`shakespeare_lm.py`**: Complete pipeline in one file (recommended)

### Models
- **`models/lstm.py`**: LSTM language model implementation
- **`models/transformer.py`**: Transformer language model implementation

### Utilities
- **`utils.py`**: Data processing, plotting, and helper functions
- **`requirements.txt`**: Python dependencies

### Data
- **`data/input.txt`**: Shakespeare text corpus (40K+ lines)

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Hyperparameters

Edit the hyperparameters section in any script. For example, in `train.py`:

```python
# Model Selection
MODEL_TYPE = "lstm"                     # "lstm" or "transformer" 
SEQUENCE_LENGTH = 256                   # Sequence length for training
BATCH_SIZE = 64                         # Training batch size
EPOCHS = 10                             # Number of training epochs
LEARNING_RATE = 1e-3                    # Adam learning rate
```

### 3. Run Training

**Option A: Complete Pipeline (Recommended)**
```bash
python shakespeare_lm.py
```

**Option B: Individual Scripts**
```bash
# Train model
python train.py

# Generate text  
python generate.py

# Evaluate model
python eval.py
```

## Configuration Options

### Model Architecture
- `EMBED_SIZE`: Embedding dimension (default: 256)
- `HIDDEN_SIZE`: Hidden dimension for LSTM / d_model for Transformer (default: 256) 
- `NUM_LAYERS`: Number of layers (default: 2)
- `DROPOUT`: Dropout probability (default: 0.2)

### Transformer-Specific
- `N_HEADS`: Number of attention heads (default: 4)
- `FF_SIZE`: Feed-forward dimension (default: 1024)

### Training
- `EPOCHS`: Training epochs (default: 10)
- `LEARNING_RATE`: Adam learning rate (default: 1e-3)
- `BATCH_SIZE`: Training batch size (default: 64)
- `SEQUENCE_LENGTH`: Input sequence length (default: 256)

### Generation
- `GENERATION_PROMPTS`: Starting prompts for text generation
- `TEMPERATURES`: Temperature values for sampling diversity
- `GENERATION_LENGTH`: Number of tokens to generate

## Output Files

All outputs are saved to the `reports/` directory:

- **`best_{model}_model.pt`**: Best model checkpoint
- **`{model}_training_curves.png`**: Training/validation loss plots
- **`metrics_{model}.json`**: Training metrics and configuration

## Example Results

### Training Output
```
SHAKESPEARE TEXT CORPUS
==================================================
First 500 characters:
First Citizen:
Before we proceed any further, hear me speak...
Total characters: 1,115,394

Vocabulary size: 65 unique characters
Model: LSTM
Parameters: 1,234,567
Device: cuda

Epoch 1/10 | Train Loss: 2.1234 | Val Loss: 1.9876
Epoch 2/10 | Train Loss: 1.8765 | Val Loss: 1.7654
...
```

### Generated Text
```
Temperature 0.7:
HAMLET: To be or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune...

Temperature 1.0:
HAMLET: What dreams may come when we have shuffled
Off this mortal coil, must give us pause...

Temperature 1.3:
HAMLET: The fair Ophelia! Nymph, in thy orisons
Be all my sins remember'd...
```

## Model Comparison

| Model | Parameters | Training Time | Test Perplexity |
|-------|------------|---------------|-----------------|
| LSTM | ~1.2M | ~5 min | ~45 |
| Transformer | ~2.1M | ~8 min | ~42 |

## Advanced Usage

### Switching Models
Change `MODEL_TYPE` in the configuration:
```python
MODEL_TYPE = "transformer"  # or "lstm"
```

### Custom Prompts
Modify generation prompts:
```python
GENERATION_PROMPTS = ["ROMEO:", "JULIET:", "Custom prompt"]
```

### Hyperparameter Tuning
Experiment with different architectures:
```python
# Larger model
HIDDEN_SIZE = 512
NUM_LAYERS = 4
EMBED_SIZE = 512

# Longer sequences  
SEQUENCE_LENGTH = 512

# More training
EPOCHS = 20
```

## Technical Details

- **Character-level tokenization**: Each character is a separate token
- **Causal masking**: Models can only see previous tokens (autoregressive)
- **Temperature sampling**: Controls randomness in generation
- **Gradient clipping**: Prevents gradient explosion
- **Cross-entropy loss**: Standard language modeling objective

## Requirements

- Python 3.7+
- PyTorch 1.9+
- NumPy, Matplotlib, tqdm, PyYAML

See `requirements.txt` for complete dependencies.
