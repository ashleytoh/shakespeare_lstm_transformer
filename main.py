#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import os

# Import our custom modules
from config import *
from data_utils import set_seed, load_shakespeare_text, build_vocabulary, encode_text, split_data
from evaluation_utils import run_ablation_studies, save_comprehensive_results


def main():
    """Main function - complete pipeline with comprehensive evaluation and ablation studies"""
    print("=" * 80)
    print("DSA4213 Assignment 2 - Shakespeare Language Model")
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
    print("\n" + "=" * 60)
    print("DATA LOADING AND PREPROCESSING")
    print("=" * 60)
    
    text = load_shakespeare_text(DATA_PATH, reduction_factor=0.7)
    char_to_idx, idx_to_char, vocab_size = build_vocabulary(text)
    data = encode_text(text, char_to_idx)
    
    # Create train/val/test splits
    train_data, val_data, test_data = split_data(data, TRAIN_SPLIT, VAL_SPLIT)
    
    # Calculate splits for reporting
    full_dataset_size = len(data)
    reported_train = int(full_dataset_size * TRAIN_SPLIT)
    reported_val = int(full_dataset_size * VAL_SPLIT)
    reported_test = full_dataset_size - reported_train - reported_val
    
    print(f"\nData splits:")
    print(f"  Train: {reported_train:,} characters")
    print(f"  Validation: {reported_val:,} characters")
    print(f"  Test: {reported_test:,} characters")
    
    # Run ablation studies
    print(f"\n{'='*80}")
    print("RUNNING ABLATION STUDIES")
    print(f"{'='*80}")
    
    ablation_results = run_ablation_studies(
        char_to_idx, idx_to_char, vocab_size,
        train_data, val_data, test_data, device
    )
    
    # Save comprehensive results
    save_comprehensive_results(ablation_results)
    
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print(f"{'='*80}")
    print(f"All results saved to: {OUTPUT_DIR}/")
    print("Files generated:")
    print("  - Training curves for each experiment")
    print("  - Text generation samples (1000 tokens each)")
    print("  - Comprehensive evaluation report (JSON)")
    print("  - Best model checkpoints")
    print("  - Ablation study comparisons")
    
    print(f"\nKey findings summary:")
    for study_name, results in ablation_results.items():
        best_result = min(results, key=lambda x: x['test_loss'])
        print(f"  {study_name}: Best configuration was '{best_result['experiment_name']}' with test perplexity {best_result['test_perplexity']:.2f}")


if __name__ == "__main__":
    main()
