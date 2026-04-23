#!/usr/bin/env python3
"""
CT Training Script

Supports both 2D (slice-by-slice) and 3D (patch-based) training.
Uses same training protocol as CXR for fair comparison.

Usage:
    # 2D mode (uses standard 2D UNet)
    python scripts/train_ct.py --model unet --mode 2d
    
    # 3D mode (uses 3D UNet)
    python scripts/train_ct.py --model unet3d --mode 3d
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import get_model, UNet3D
from datasets.ct_dataset import create_ct_data_loaders, CTDataset2D, CTDataset3D
from utils.losses import get_loss_function
from utils.metrics import SegmentationMetrics
from engine.trainer import deep_merge, set_seed, EarlyStopping


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


def load_config(model_config_path=None, base_path=None):
    """Load config with inheritance."""
    config = {}
    
    if base_path is None:
        base_path = project_root / 'configs' / 'base.yaml'
    
    if Path(base_path).exists():
        with open(base_path, 'r') as f:
            config = yaml.safe_load(f) or {}
    
    if model_config_path and Path(model_config_path).exists():
        with open(model_config_path, 'r') as f:
            model_cfg = yaml.safe_load(f) or {}
        config = deep_merge(config, model_cfg)
    
    return config


def get_output_dirs(config, mode):
    """Generate output directory for CT training."""
    base_dir = str(config.get('output', {}).get('base_dir', 'outputs'))
    model_name = str(config.get('model', {}).get('name', 'unknown'))
    seed = int(config.get('training', {}).get('seed', 42))
    run_id = config.get('output', {}).get('run_id')
    
    if run_id is None:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        run_id = f"{timestamp}_seed{seed}"
    
    # CT outputs go to outputs/{dataset}_{mode}/{model}/{run_id}/
    # dataset_name comes from config (e.g., 'luna16' or 'ct')
    dataset_base = config.get('dataset_name', 'ct')
    dataset_name = f"{dataset_base}_{mode}"
    output_path = Path(base_dir) / dataset_name / model_name / run_id
    
    dirs = {
        'output_dir': output_path,
        'log_dir': output_path / 'logs',
        'save_dir': output_path / 'checkpoints',
        'results_dir': output_path / 'results',
        'vis_dir': output_path / 'visualizations'
    }
    
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    
    return dirs


def save_run_metadata(output_dir, config, args):
    """Save run metadata."""
    output_dir = Path(output_dir)
    
    # Config
    config_path = output_dir / 'config_resolved.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    # Metadata
    dataset_name = config.get('dataset_name', 'ct')
    meta = {
        'dataset_train': f'{dataset_name}_{args.mode}',
        'mode': args.mode,
        'model': config['model']['name'],
        'seed': config['training']['seed'],
        'run_id': config['output'].get('run_id'),
        'index_path': config.get('index_path', 'data/index_ct.csv'),
        'git_commit': get_git_commit(),
        'timestamp': datetime.now().isoformat(),
    }
    
    meta_path = output_dir / 'run_meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    
    # Pip freeze
    pip_freeze = get_pip_freeze()
    if pip_freeze:
        freeze_path = output_dir / 'pip_freeze.txt'
        with open(freeze_path, 'w') as f:
            f.write(pip_freeze)
    
    print(f"Saved run metadata to {output_dir}")


class CTTrainer:
    """CT Trainer - supports 2D and 3D modes."""
    
    def __init__(self, model, config, mode='2d', device=None):
        self.model = model
        self.config = config
        self.mode = mode
        
        # Set seed
        seed = int(config.get('training', {}).get('seed', 42))
        set_seed(seed)
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Output dirs
        self.dirs = get_output_dirs(config, mode)
        
        print(f"Output directory: {self.dirs['output_dir']}")
        print(f"Mode: {mode}")
        
        # Data loaders
        self._create_data_loaders()
        
        # Loss - same as CXR for fair comparison
        # For CT we use simple Dice+BCE, pos_weight=1 (CT masks are sparse)
        self.criterion = get_loss_function(config, pos_weight=1.0)
        
        # Optimizer - same as CXR
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        
        # Metrics
        self.train_metrics = SegmentationMetrics()
        self.val_metrics = SegmentationMetrics()
        
        # Early stopping
        early_cfg = config.get('early_stopping', {})
        self.early_stopping = EarlyStopping(
            patience=int(early_cfg.get('patience', 10)),
            min_delta=float(early_cfg.get('min_delta', 0.001)),
            mode=str(early_cfg.get('mode', 'max'))
        )
        
        # History
        self.history = {
            'train_loss': [], 'train_dice': [],
            'val_loss': [], 'val_dice': [],
            'learning_rates': []
        }
        
        self.writer = None
        
        print(f"Training on device: {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
    
    def _create_data_loaders(self):
        """Create CT data loaders."""
        data_root = self.config.get('data_root', '.')
        index_path = self.config.get('index_path', 'data/index_ct.csv')
        batch_size = int(self.config['training']['batch_size'])
        num_workers = int(self.config.get('num_workers', 0))
        ct_cfg = self.config.get('ct', {})
        
        print(f"Loading data from index: {index_path}")
        
        if self.mode == '2d':
            # 2D slice-by-slice
            image_size = self.config.get('dataset', {}).get('image_size', [512, 512])
            if isinstance(image_size, list):
                image_size = [int(x) for x in image_size]
            
            dataset_kwargs = {
                'data_root': data_root,
                'index_path': index_path,
                'image_size': image_size,
                'window_center': float(ct_cfg.get('window_center', -600)),
                'window_width': float(ct_cfg.get('window_width', 1500)),
            }
            
            self.train_dataset = CTDataset2D(**dataset_kwargs, split='train')
            self.val_dataset = CTDataset2D(**dataset_kwargs, split='val')
            self.test_dataset = CTDataset2D(**dataset_kwargs, split='test')
            
        else:  # 3d
            patch_size = ct_cfg.get('patch_size', [128, 128, 32])
            if isinstance(patch_size, list):
                patch_size = [int(x) for x in patch_size]
            
            dataset_kwargs = {
                'data_root': data_root,
                'index_path': index_path,
                'patch_size': patch_size,
                'window_center': float(ct_cfg.get('window_center', -600)),
                'window_width': float(ct_cfg.get('window_width', 1500)),
                'patches_per_volume': int(ct_cfg.get('patches_per_volume', 4)),
            }
            
            self.train_dataset = CTDataset3D(**dataset_kwargs, split='train')
            self.val_dataset = CTDataset3D(**dataset_kwargs, split='val')
            self.test_dataset = CTDataset3D(**dataset_kwargs, split='test')
        
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=torch.cuda.is_available()
        )
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=torch.cuda.is_available()
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=torch.cuda.is_available()
        )
    
    def _get_optimizer(self):
        """Get optimizer - same as CXR."""
        opt_name = self.config['training'].get('optimizer', 'adam').lower()
        lr = float(self.config['training']['learning_rate'])
        wd = float(self.config['training'].get('weight_decay', 1e-4))
        
        if opt_name == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == 'adamw':
            return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
        else:
            return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
    
    def _get_scheduler(self):
        """Get scheduler - same as CXR."""
        sched_cfg = self.config.get('scheduler', {})
        if not sched_cfg:
            return None
        
        sched_type = sched_cfg.get('type', 'reduce_on_plateau')
        
        if sched_type == 'reduce_on_plateau':
            factor = float(sched_cfg.get('factor', 0.5))
            patience = int(sched_cfg.get('patience', 3))
            min_lr = float(sched_cfg.get('min_lr', 1e-6))
            threshold = float(sched_cfg.get('threshold', 1e-3))
            
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max',
                factor=factor,
                patience=patience,
                min_lr=min_lr,
                threshold=threshold,
                verbose=True
            )
        elif sched_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=int(self.config['training']['num_epochs'])
            )
        return None
    
    def train_epoch(self):
        """Train one epoch."""
        self.model.train()
        self.train_metrics.reset()
        
        running_loss = 0.0
        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            self.train_metrics.update(outputs, masks)
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_metrics = self.train_metrics.compute()
        
        return epoch_loss, epoch_metrics
    
    def validate_epoch(self):
        """Validate one epoch."""
        self.model.eval()
        self.val_metrics.reset()
        
        running_loss = 0.0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation', leave=False)
            
            for batch in pbar:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                running_loss += loss.item()
                self.val_metrics.update(outputs, masks)
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_metrics = self.val_metrics.compute()
        
        return epoch_loss, epoch_metrics
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'mode': self.mode
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        last_path = self.dirs['save_dir'] / 'last.pt'
        torch.save(checkpoint, last_path)
        
        if is_best:
            best_path = self.dirs['save_dir'] / 'best.pt'
            torch.save(checkpoint, best_path)
            print(f"Best model saved (Dice={metrics['dice']:.4f})")
    
    def train(self, num_epochs=None):
        """Train the model."""
        if num_epochs is None:
            num_epochs = int(self.config['training']['num_epochs'])
        
        best_dice = 0.0
        
        if self.writer is None:
            self.writer = SummaryWriter(str(self.dirs['log_dir']))
        
        print(f"\nStarting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            train_loss, train_metrics = self.train_epoch()
            val_loss, val_metrics = self.validate_epoch()
            
            # Scheduler step
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['dice'])
                else:
                    self.scheduler.step()
            
            # History
            self.history['train_loss'].append(train_loss)
            self.history['train_dice'].append(train_metrics['dice'])
            self.history['val_loss'].append(val_loss)
            self.history['val_dice'].append(val_metrics['dice'])
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # TensorBoard
            if self.writer:
                self.writer.add_scalars('Loss', {'Train': train_loss, 'Val': val_loss}, epoch)
                self.writer.add_scalars('Dice', {'Train': train_metrics['dice'], 'Val': val_metrics['dice']}, epoch)
            
            # Save checkpoint
            is_best = val_metrics['dice'] > best_dice
            if is_best:
                best_dice = val_metrics['dice']
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Early stopping
            if self.early_stopping(val_metrics['dice']):
                print(f"Early stopping at epoch {epoch}")
                break
            
            # Log
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train - Loss: {train_loss:.4f}, Dice: {train_metrics['dice']:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Dice: {val_metrics['dice']:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        if self.writer:
            self.writer.close()
        
        # Save history
        import pandas as pd
        df = pd.DataFrame({
            'epoch': range(1, len(self.history['train_loss']) + 1),
            'train_loss': self.history['train_loss'],
            'train_dice': self.history['train_dice'],
            'val_loss': self.history['val_loss'],
            'val_dice': self.history['val_dice'],
            'learning_rate': self.history['learning_rates']
        })
        df.to_csv(self.dirs['results_dir'] / 'metrics.csv', index=False)
        
        print(f"\nTraining completed! Best Dice: {best_dice:.4f}")
        print(f"Results saved to: {self.dirs['output_dir']}")
        
        return self.history


def main():
    parser = argparse.ArgumentParser(description='Train CT segmentation model')
    
    parser.add_argument('--model', type=str, default='unet',
                        choices=['unet', 'attention_unet', 'se_unet', 'cbam_unet', 'unet3d'],
                        help='Model name')
    parser.add_argument('--mode', type=str, default='2d', choices=['2d', '3d'],
                        help='Training mode: 2d (slice) or 3d (patch)')
    parser.add_argument('--config', type=str, help='Path to model config')
    
    # Training overrides
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    parser.add_argument('--data-root', type=str, default=str(project_root))
    parser.add_argument('--index-path', type=str, default='data/index_ct.csv',
                        help='Path to index CSV (relative to data_root)')
    parser.add_argument('--dataset-name', type=str, default=None,
                        help='Dataset name for output directory (default: auto from index)')
    parser.add_argument('--dry-run', action='store_true')
    
    args = parser.parse_args()
    
    # Force 3D model for 3D mode
    if args.mode == '3d' and args.model != 'unet3d':
        print(f"Warning: 3D mode requires unet3d model, switching from {args.model} to unet3d")
        args.model = 'unet3d'
    
    # Load config
    if args.config:
        model_config_path = args.config
    else:
        model_config_path = project_root / 'configs' / f'{args.model}.yaml'
    
    config = load_config(model_config_path)
    
    # Apply CLI overrides
    if args.epochs:
        config['training']['num_epochs'] = int(args.epochs)
    if args.batch_size:
        config['training']['batch_size'] = int(args.batch_size)
    if args.lr:
        config['training']['learning_rate'] = float(args.lr)
    if args.seed:
        config['training']['seed'] = int(args.seed)
    
    config['data_root'] = args.data_root
    config['index_path'] = args.index_path
    
    # Dataset name for output directory
    if args.dataset_name:
        config['dataset_name'] = args.dataset_name
    else:
        # Auto-detect from index path: data/index_luna16.csv -> luna16
        index_stem = Path(args.index_path).stem  # index_luna16 or index_ct
        if index_stem.startswith('index_'):
            config['dataset_name'] = index_stem[6:]  # luna16 or ct
        else:
            config['dataset_name'] = 'ct'
    
    # Generate run ID
    seed = int(config['training']['seed'])
    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_seed{seed}"
    config['output']['run_id'] = run_id
    
    # Print config
    print("="*60)
    print("CT Training Configuration")
    print("="*60)
    print(f"Model: {config['model']['name']}")
    print(f"Mode: {args.mode}")
    print(f"Dataset: {config['dataset_name']}")
    print(f"Index: {config['index_path']}")
    print(f"Epochs: {config['training']['num_epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Seed: {seed}")
    print(f"Run ID: {run_id}")
    print("="*60)
    
    if args.dry_run:
        print("\nDry run - exiting")
        return
    
    # Create model
    print("\nCreating model...")
    if args.mode == '3d':
        model = UNet3D(
            in_channels=config['model'].get('in_channels', 1),
            out_channels=config['model'].get('out_channels', 1),
            features=config['model'].get('features', [32, 64, 128, 256]),
            dropout=config['model'].get('dropout', 0.0)
        )
    else:
        model = get_model(config)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = CTTrainer(model, config, mode=args.mode)
    
    # Save metadata
    save_run_metadata(trainer.dirs['output_dir'], config, args)
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()
