import os
import time
import json
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import yaml

from utils.losses import get_loss_function
from utils.metrics import SegmentationMetrics
from datasets import create_data_loaders


def get_output_dirs(config):
    """
    Generate output directory structure: 
    outputs/{dataset}/{model}/{timestamp}_seed{seed}/
    
    Returns dict with log_dir, save_dir, results_dir, vis_dir
    """
    base_dir = config.get('output', {}).get('base_dir', 'outputs')
    dataset_name = config.get('data', {}).get('dataset_name', 'unknown')
    model_name = config.get('model', {}).get('name', 'unknown')
    seed = config.get('training', {}).get('seed', 42)
    run_id = config.get('output', {}).get('run_id')
    
    # Auto-generate run_id with timestamp + seed
    if run_id is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_id = f"{timestamp}_seed{seed}"
    
    # Build output path
    output_path = Path(base_dir) / dataset_name / model_name / run_id
    
    dirs = {
        'output_dir': output_path,
        'log_dir': output_path / 'logs',
        'save_dir': output_path / 'checkpoints',
        'results_dir': output_path / 'results',
        'vis_dir': output_path / 'visualizations'
    }
    
    # Create directories
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    
    return dirs


def set_seed(seed):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """Early stopping mechanism."""
    
    def __init__(self, patience=7, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'max':
            self.compare = lambda score, best: score > best + self.min_delta
        else:
            self.compare = lambda score, best: score < best - self.min_delta
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.compare(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class Trainer:
    """Model trainer with dynamic output directory support."""
    
    def __init__(self, model, config, device=None):
        self.model = model
        self.config = config
        
        # Set seed for reproducibility
        seed = config.get('training', {}).get('seed', 42)
        set_seed(seed)
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Setup output directories
        self.dirs = get_output_dirs(config)
        self.log_dir = self.dirs['log_dir']
        self.save_dir = self.dirs['save_dir']
        self.results_dir = self.dirs['results_dir']
        self.vis_dir = self.dirs['vis_dir']
        
        print(f"Output directory: {self.dirs['output_dir']}")
        
        # Save config snapshot (critical for reproducibility)
        self._save_config_snapshot()
        
        # Create data loaders
        self.train_loader, self.val_loader, self.test_loader, self.pos_weight = create_data_loaders(config)
        
        # Loss function
        self.criterion = get_loss_function(config, self.pos_weight)
        
        # Optimizer & LR scheduler
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        
        # Metric trackers
        self.train_metrics = SegmentationMetrics()
        self.val_metrics = SegmentationMetrics()
        
        # Early stopping
        early_stop_config = config.get('early_stopping', {})
        self.early_stopping = EarlyStopping(
            patience=early_stop_config.get('patience', 10),
            min_delta=early_stop_config.get('min_delta', 0.001),
            mode=early_stop_config.get('mode', 'max')
        )
        
        # TensorBoard writer
        self.writer = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_dice': [],
            'val_loss': [],
            'val_dice': [],
            'learning_rates': []
        }
        
        print(f"Training on device: {self.device}")
        print(f"Seed: {seed}")
        print(f"Dataset sizes - Train: {len(self.train_loader.dataset)}, "
              f"Val: {len(self.val_loader.dataset)}, Test: {len(self.test_loader.dataset)}")
        print(f"Positive weight: {self.pos_weight:.3f}")
    
    def _save_config_snapshot(self):
        """Save resolved config to output directory for reproducibility."""
        # Save as YAML
        config_yaml_path = self.dirs['output_dir'] / 'config_resolved.yaml'
        with open(config_yaml_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        
        # Also save as JSON for easy parsing
        config_json_path = self.dirs['output_dir'] / 'config_resolved.json'
        with open(config_json_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"Config saved to {config_yaml_path}")
    
    def _get_optimizer(self):
        """Get optimizer."""
        optimizer_name = self.config['training'].get('optimizer', 'adam').lower()
        lr = self.config['training']['learning_rate']
        weight_decay = self.config['training'].get('weight_decay', 1e-4)
        
        if optimizer_name == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        elif optimizer_name == 'adamw':
            return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _get_scheduler(self):
        """Get learning rate scheduler."""
        scheduler_config = self.config.get('scheduler', {})
        if not scheduler_config:
            return None

        scheduler_type = scheduler_config.get('type', 'reduce_on_plateau')

        if scheduler_type == 'reduce_on_plateau':
            factor = float(scheduler_config.get('factor', 0.5))
            patience = int(scheduler_config.get('patience', 4))
            min_lr = float(scheduler_config.get('min_lr', 1e-5))
            threshold = float(scheduler_config.get('threshold', 1e-3))

            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=factor,
                patience=patience,
                min_lr=min_lr,
                threshold=threshold,
                verbose=True
            )

        elif scheduler_type == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=int(scheduler_config.get('step_size', 20)),
                gamma=float(scheduler_config.get('gamma', 0.1)),
            )

        elif scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=int(self.config['training']['num_epochs'])
            )
        else:
            return None

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()
        
        running_loss = 0.0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            self.train_metrics.update(outputs, masks)
            
            current_loss = running_loss / (batch_idx + 1)
            pbar.set_postfix({'Loss': f'{current_loss:.4f}'})
        
        epoch_loss = running_loss / num_batches
        epoch_metrics = self.train_metrics.compute()
        
        return epoch_loss, epoch_metrics
    
    def validate_epoch(self):
        """Validate for one epoch."""
        self.model.eval()
        self.val_metrics.reset()
        
        running_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation', leave=False)
            
            for batch_idx, batch in enumerate(pbar):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                running_loss += loss.item()
                self.val_metrics.update(outputs, masks)
                
                current_loss = running_loss / (batch_idx + 1)
                pbar.set_postfix({'Loss': f'{current_loss:.4f}'})
        
        epoch_loss = running_loss / num_batches
        epoch_metrics = self.val_metrics.compute()
        
        return epoch_loss, epoch_metrics
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save last checkpoint
        last_path = self.save_dir / 'last.pt'
        torch.save(checkpoint, last_path)
        
        # Save best model
        if is_best:
            best_path = self.save_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            print(f"Best model saved to {best_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint['epoch'], checkpoint['metrics']
    
    def train(self, num_epochs=None, resume_from=None):
        """Train the model."""
        if num_epochs is None:
            num_epochs = self.config['training']['num_epochs']
        
        start_epoch = 0
        best_dice = 0.0
        
        if resume_from:
            start_epoch, metrics = self.load_checkpoint(resume_from)
            best_dice = metrics.get('dice', 0.0)
            print(f"Resumed training from epoch {start_epoch}")
        
        if self.writer is None:
            self.writer = SummaryWriter(str(self.log_dir))
        
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()
            
            train_loss, train_metrics = self.train_epoch()
            val_loss, val_metrics = self.validate_epoch()
            
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['dice'])
                else:
                    self.scheduler.step()
            
            self.history['train_loss'].append(train_loss)
            self.history['train_dice'].append(train_metrics['dice'])
            self.history['val_loss'].append(val_loss)
            self.history['val_dice'].append(val_metrics['dice'])
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            if self.writer:
                self.writer.add_scalars('Loss', {
                    'Train': train_loss,
                    'Validation': val_loss
                }, epoch)
                
                self.writer.add_scalars('Dice Score', {
                    'Train': train_metrics['dice'],
                    'Validation': val_metrics['dice']
                }, epoch)
                
                self.writer.add_scalar('Learning Rate', 
                                     self.optimizer.param_groups[0]['lr'], epoch)
            
            is_best = val_metrics['dice'] > best_dice
            if is_best:
                best_dice = val_metrics['dice']
            
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            if self.early_stopping(val_metrics['dice']):
                print(f"Early stopping triggered at epoch {epoch}")
                break
            
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch+1}/{num_epochs} - {epoch_time:.2f}s")
            print(f"  Train - Loss: {train_loss:.4f}, Dice: {train_metrics['dice']:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            if (epoch + 1) % 10 == 0:
                print(f"  Detailed Val Metrics:")
                print(f"    Precision: {val_metrics['precision']:.4f}")
                print(f"    Recall: {val_metrics['recall']:.4f}")
                print(f"    Accuracy: {val_metrics['accuracy']:.4f}")
        
        if self.writer:
            self.writer.close()
        
        # Save training history as CSV (easier for paper writing)
        self._save_metrics_csv()
        
        # Save as JSON too
        history_path = self.results_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"Training completed! Best validation Dice: {best_dice:.4f}")
        print(f"Results saved to: {self.dirs['output_dir']}")
        return self.history
    
    def _save_metrics_csv(self):
        """Save training metrics as CSV."""
        import pandas as pd
        
        df = pd.DataFrame({
            'epoch': range(1, len(self.history['train_loss']) + 1),
            'train_loss': self.history['train_loss'],
            'train_dice': self.history['train_dice'],
            'val_loss': self.history['val_loss'],
            'val_dice': self.history['val_dice'],
            'learning_rate': self.history['learning_rates']
        })
        
        csv_path = self.results_dir / 'metrics.csv'
        df.to_csv(csv_path, index=False)
        print(f"Metrics saved to {csv_path}")
    
    def plot_training_curves(self, save_path=None):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss', alpha=0.8)
        axes[0, 0].plot(self.history['val_loss'], label='Validation Loss', alpha=0.8)
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(self.history['train_dice'], label='Train Dice', alpha=0.8)
        axes[0, 1].plot(self.history['val_dice'], label='Validation Dice', alpha=0.8)
        axes[0, 1].set_title('Training and Validation Dice Score')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        axes[1, 0].plot(self.history['learning_rates'], alpha=0.8)
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(self.history['val_loss'], label='Val Loss (scaled)', alpha=0.8)
        axes[1, 1].plot(self.history['val_dice'], label='Val Dice', alpha=0.8)
        axes[1, 1].set_title('Validation Metrics Comparison')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.vis_dir / 'training_curves.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def deep_merge(base, override):
    """Deep merge override into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path, dataset_config=None, base_path=None):
    """
    Load config with multi-layer inheritance:
    base.yaml -> dataset config -> model config
    
    Args:
        config_path: path to model config (e.g., configs/unet.yaml)
        dataset_config: optional path to dataset config (e.g., configs/datasets/kaggle.yaml)
        base_path: optional path to base config
    """
    config_path = Path(config_path)
    config = {}
    
    # 1. Load base config
    if base_path is None:
        # Try to find base.yaml in same directory or parent
        base_candidates = [
            config_path.parent / 'base.yaml',
            config_path.parent.parent / 'base.yaml',
        ]
        for candidate in base_candidates:
            if candidate.exists():
                base_path = candidate
                break
    
    if base_path and Path(base_path).exists():
        with open(base_path, 'r') as f:
            config = yaml.safe_load(f) or {}
    
    # 2. Load dataset config if provided
    if dataset_config:
        dataset_config = Path(dataset_config)
        if dataset_config.exists():
            with open(dataset_config, 'r') as f:
                ds_cfg = yaml.safe_load(f) or {}
            config = deep_merge(config, ds_cfg)
    
    # 3. Load model config
    with open(config_path, 'r') as f:
        model_cfg = yaml.safe_load(f) or {}
    
    config = deep_merge(config, model_cfg)
    
    return config


def train_model_from_config(config_path, model_class, resume_from=None):
    """Train a model from a YAML config."""
    config = load_config(config_path)
    
    # Build model
    model_cfg = config.get('model', {})
    model = model_class(
        in_channels=model_cfg.get('in_channels', 1),
        out_channels=model_cfg.get('out_channels', 1),
        features=model_cfg.get('features', [64, 128, 256, 512]),
        dropout=model_cfg.get('dropout', 0.0),
        bilinear=model_cfg.get('bilinear', True)
    )
    
    # Build trainer
    trainer = Trainer(model, config)
    
    # Train
    history = trainer.train(resume_from=resume_from)
    
    return trainer, history
