#!/usr/bin/env python3
"""
Unified Training Script

Usage:
    python scripts/train.py --model unet --dataset kaggle
    python scripts/train.py --model segformer --dataset montgomery --epochs 50
    python scripts/train.py --config configs/unet.yaml --dataset-config configs/datasets/kaggle.yaml

Config priority: base.yaml < dataset.yaml < model.yaml < CLI args
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import torch

from models import get_model
from engine.trainer import Trainer, deep_merge


def get_git_commit():
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, cwd=project_root
        )
        return result.stdout.strip()[:8] if result.returncode == 0 else None
    except:
        return None


def get_pip_freeze():
    """Get pip freeze output."""
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'freeze'],
            capture_output=True, text=True
        )
        return result.stdout if result.returncode == 0 else None
    except:
        return None


def load_config(model_config_path=None, dataset_config_path=None, base_path=None):
    """
    Load config with multi-layer inheritance.
    Priority: base.yaml < dataset.yaml < model.yaml
    """
    config = {}
    
    # 1. Load base config
    if base_path is None:
        base_path = project_root / 'configs' / 'base.yaml'
    
    if Path(base_path).exists():
        with open(base_path, 'r') as f:
            config = yaml.safe_load(f) or {}
    
    # 2. Load dataset config
    if dataset_config_path and Path(dataset_config_path).exists():
        with open(dataset_config_path, 'r') as f:
            ds_cfg = yaml.safe_load(f) or {}
        config = deep_merge(config, ds_cfg)
    
    # 3. Load model config
    if model_config_path and Path(model_config_path).exists():
        with open(model_config_path, 'r') as f:
            model_cfg = yaml.safe_load(f) or {}
        config = deep_merge(config, model_cfg)
    
    return config


def apply_cli_overrides(config, args):
    """Apply CLI argument overrides to config."""
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.seed:
        config['training']['seed'] = args.seed
    
    return config


def generate_run_id(seed):
    """Generate unique run ID: YYYYMMDD-HHMMSS_seed{seed}"""
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    return f"{timestamp}_seed{seed}"


def save_run_metadata(output_dir, config, args):
    """Save run metadata for reproducibility."""
    output_dir = Path(output_dir)
    
    # 1. Save resolved config
    config_path = output_dir / 'config_resolved.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    # 2. Save run metadata
    meta = {
        'dataset_train': config['data'].get('dataset_name', 'unknown'),
        'dataset_filter': config['data'].get('dataset_filter'),
        'model': config['model']['name'],
        'seed': config['training']['seed'],
        'run_id': config['output'].get('run_id'),
        'config_resolved_path': str(config_path),
        'git_commit': get_git_commit(),
        'timestamp': datetime.now().isoformat(),
    }
    
    meta_path = output_dir / 'run_meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    
    # 3. Save pip freeze
    pip_freeze = get_pip_freeze()
    if pip_freeze:
        freeze_path = output_dir / 'pip_freeze.txt'
        with open(freeze_path, 'w') as f:
            f.write(pip_freeze)
    
    print(f"Saved run metadata to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train segmentation model')
    
    # Model and dataset
    parser.add_argument('--model', type=str, choices=['unet', 'attention_unet', 'se_unet', 'cbam_unet', 'segformer'],
                        help='Model name')
    parser.add_argument('--dataset', type=str, choices=['kaggle', 'montgomery', 'shenzhen', 'all'],
                        help='Dataset name')
    
    # Config files (alternative to --model/--dataset)
    parser.add_argument('--config', type=str, help='Path to model config')
    parser.add_argument('--dataset-config', type=str, help='Path to dataset config')
    parser.add_argument('--base-config', type=str, help='Path to base config')
    
    # Training overrides
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Other
    parser.add_argument('--data-root', type=str, default=str(project_root), help='Data root directory')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--dry-run', action='store_true', help='Print config and exit')
    
    args = parser.parse_args()
    
    # Determine config paths
    if args.config:
        model_config_path = args.config
    elif args.model:
        model_config_path = project_root / 'configs' / f'{args.model}.yaml'
    else:
        parser.error('Either --model or --config is required')
    
    if args.dataset_config:
        dataset_config_path = args.dataset_config
    elif args.dataset and args.dataset != 'all':
        dataset_config_path = project_root / 'configs' / 'datasets' / f'{args.dataset}.yaml'
    else:
        dataset_config_path = None
    
    base_config_path = args.base_config or (project_root / 'configs' / 'base.yaml')
    
    # Load and merge configs
    config = load_config(model_config_path, dataset_config_path, base_config_path)
    
    # Apply CLI overrides
    config = apply_cli_overrides(config, args)
    
    # Set data root
    config['data_root'] = args.data_root
    os.environ['DATA_ROOT'] = args.data_root
    
    # Generate run ID
    seed = config['training']['seed']
    run_id = generate_run_id(seed)
    config['output']['run_id'] = run_id
    
    # Print config summary
    print("="*60)
    print("Training Configuration")
    print("="*60)
    print(f"Model: {config['model']['name']}")
    print(f"Dataset: {config['data'].get('dataset_name', 'all')}")
    print(f"Epochs: {config['training']['num_epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Seed: {seed}")
    print(f"Run ID: {run_id}")
    print("="*60)
    
    if args.dry_run:
        print("\nFull config:")
        print(yaml.dump(config, default_flow_style=False))
        return
    
    # Create model
    print("\nCreating model...")
    model = get_model(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(model, config)
    
    # Save run metadata
    save_run_metadata(trainer.dirs['output_dir'], config, args)
    
    # Train
    print("\nStarting training...")
    history = trainer.train(resume_from=args.resume)
    
    # Plot training curves
    trainer.plot_training_curves()
    
    print("\nTraining completed!")
    print(f"Results saved to: {trainer.dirs['output_dir']}")


if __name__ == '__main__':
    main()
