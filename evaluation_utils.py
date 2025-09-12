#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation and Experiment Management Utilities
Handles ablation studies, result comparison, and comprehensive evaluation
"""

import json
import os
import torch


def run_single_experiment(config_override, study_name, experiment_name, 
                         char_to_idx, idx_to_char, vocab_size, 
                         train_data, val_data, test_data, device):
    """
    Run a single experiment with configuration override
    
    Args:
        config_override: Dictionary with parameter overrides
        study_name: Name of the study
        experiment_name: Name of the experiment
        char_to_idx, idx_to_char: Character mappings
        vocab_size: Size of vocabulary
        train_data, val_data, test_data: Data splits
        device: PyTorch device
    
    Returns:
        dict: Experiment results
    """
    # Import here to avoid circular imports
    import config
    from data_utils import create_data_loaders
    from models import create_model
    from training_utils import train_model, evaluate_model, generate_text, plot_training_curves
    
    print(f"\n{'='*80}")
    print(f"RUNNING EXPERIMENT: {study_name} - {experiment_name}")
    print(f"{'='*80}")
    
    # Override configurations
    current_seq_len = config_override.get('sequence_length', config.SEQUENCE_LENGTH)
    current_dropout = config_override.get('dropout', config.DROPOUT)
    current_model_type = config_override.get('model_type', config.MODEL_TYPE)
    current_embed_size = config_override.get('embed_size', config.EMBED_SIZE)
    current_hidden_size = config_override.get('hidden_size', config.HIDDEN_SIZE)
    
    print(f"Configuration:")
    print(f"  Model Type: {current_model_type}")
    print(f"  Sequence Length: {current_seq_len}")
    print(f"  Dropout: {current_dropout}")
    print(f"  Embed Size: {current_embed_size}")
    print(f"  Hidden Size: {current_hidden_size}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data, val_data, test_data, current_seq_len, config.BATCH_SIZE, device
    )
    
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
    train_losses, val_losses, training_time = train_model(
        model, train_loader, val_loader, device, experiment_model_name,
        config.EPOCHS, config.LEARNING_RATE, config.GRADIENT_CLIP, config.OUTPUT_DIR
    )
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, experiment_model_name, config.OUTPUT_DIR)
    
    # Evaluate on test set
    test_loss, test_perplexity = evaluate_model(model, test_loader, device)
    
    # Generate samples with all required temperatures and save to files
    generation_samples = {}
    for temp in config.TEMPERATURES:
        print(f"\n--- Sample Generation (T={temp}) ---")
        sample_text = generate_text(
            model, "HAMLET:", char_to_idx, idx_to_char, 
            config.GENERATION_LENGTH, temp, device, current_seq_len
        )
        generation_samples[f"temp_{temp}"] = sample_text
        print(sample_text[:200] + "..." if len(sample_text) > 200 else sample_text)
        
        # Save to file
        sample_file = os.path.join(config.OUTPUT_DIR, f"{experiment_model_name}_generation_T{temp}.txt")
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
    """
    Run all configured ablation studies
    
    Args:
        char_to_idx, idx_to_char: Character mappings
        vocab_size: Size of vocabulary
        train_data, val_data, test_data: Data splits
        device: PyTorch device
    
    Returns:
        dict: All ablation study results
    """
    import config
    
    print(f"\n{'='*80}")
    print("STARTING ABLATION STUDIES")
    print(f"{'='*80}")
    
    all_results = {}
    
    for study in config.ABLATION_STUDIES:
        study_name = study['name']
        study_results = []
        
        print(f"\nABLATION STUDY: {study_name.upper()}")
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
    """
    Print comparison results for a study
    
    Args:
        study_name: Name of the study
        results: List of experiment results
    """
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


def save_comprehensive_results(ablation_results):
    """
    Save comprehensive results to JSON file
    
    Args:
        ablation_results: Dictionary of all study results
    """
    import config
    
    results_file = os.path.join(config.OUTPUT_DIR, "comprehensive_results.json")
    
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
