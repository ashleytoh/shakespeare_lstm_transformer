# Windows ARM64 Setup Guide for DSA4213 Assignment 2

## Important: Windows ARM64 Compatibility

You are running on Windows ARM64, which has specific requirements for PyTorch installation.

## Installation Steps

### 1. Install Dependencies
```bash
# The basic packages should already be installed
pip install numpy matplotlib tqdm pandas scikit-learn ipykernel PyYAML
```

### 2. Install PyTorch (ARM64 Compatible)
```bash
# Use PyTorch nightly build with ARM64 support
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

### 3. Verify Installation
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print('Setup complete!')"
```

## Running the Models

### Quick Test
```bash
# Test LSTM model
python -c "from models.lstm import LSTMLanguageModel; print('LSTM model working!')"

# Test Transformer model  
python -c "from models.transformer import TinyTransformerLM; print('Transformer model working!')"
```

### Run Training
```bash
# Run complete pipeline
python shakespeare_lm.py

# Or run individual scripts
python train.py
```

## Notes

- **CUDA Support**: Not available on ARM64, models will run on CPU
- **Performance**: CPU-only training will be slower but functional
- **Memory**: Consider reducing batch sizes if memory is limited
- **Model Size**: Start with smaller models (fewer layers/hidden units) for faster iteration

## Troubleshooting

### If PyTorch Installation Fails:
1. Try the nightly build (as shown above)
2. Consider using x64 emulation with x64 Python installation
3. Use alternative libraries like TensorFlow (better ARM64 support)

### If Memory Issues Occur:
```python
# In your script configurations, reduce:
BATCH_SIZE = 32          # Instead of 64
HIDDEN_SIZE = 128        # Instead of 256
NUM_LAYERS = 1           # Instead of 2
SEQUENCE_LENGTH = 128    # Instead of 256
```

## Alternative: TensorFlow Implementation

If PyTorch continues to cause issues, consider implementing the models in TensorFlow, which has better ARM64 support:

```bash
pip install tensorflow
```

The neural network concepts are the same, just different framework syntax.
