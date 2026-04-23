#!/usr/bin/env python3
"""
Unified Evaluation Script

Outputs:
- results_cases.csv: per-case metrics (for top/bottom analysis)
- results.csv: aggregated metrics (mean/std for paper tables)

Usage:
    # Evaluate on same dataset (in-domain)
    python scripts/evaluate.py --checkpoint outputs/.../checkpoints/best.pt
    
    # Evaluate on different dataset (cross-domain)
    python scripts/evaluate.py --checkpoint outputs/.../checkpoints/best.pt --dataset montgomery
"""

import os
import sys
import json
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
import yaml

from models import get_model
from datasets.chest_xray_dataset import ChestXrayDataset
from scipy import ndimage


def load_checkpoint(checkpoint_path):
    """Load model checkpoint and config."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    return checkpoint, config


def get_run_meta(checkpoint_path):
    """Load run metadata from checkpoint directory."""
    ckpt_dir = Path(checkpoint_path).parent
    run_dir = ckpt_dir.parent
    
    meta_path = run_dir / 'run_meta.json'
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            return json.load(f)
    return {}


def keep_largest_cc(mask):
    """Keep only the largest connected component."""
    if mask.sum() == 0:
        return mask
    
    labeled, num_features = ndimage.label(mask)
    if num_features <= 1:
        return mask
    
    # Find largest component
    component_sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
    largest_label = np.argmax(component_sizes) + 1
    
    return (labeled == largest_label).astype(mask.dtype)


def compute_metrics(pred, target, threshold=0.5):
    """Compute segmentation metrics."""
    # Binarize prediction
    pred_binary = (pred > threshold).float()
    
    # Flatten
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)
    
    # Compute metrics
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    # Dice
    dice = (2 * intersection + 1e-8) / (union + 1e-8)
    
    # IoU
    iou = (intersection + 1e-8) / (union - intersection + 1e-8)
    
    # Precision, Recall
    tp = intersection
    fp = pred_flat.sum() - tp
    fn = target_flat.sum() - tp
    
    precision = (tp + 1e-8) / (tp + fp + 1e-8)
    recall = (tp + 1e-8) / (tp + fn + 1e-8)
    
    return {
        'dice': dice.item(),
        'iou': iou.item(),
        'precision': precision.item(),
        'recall': recall.item(),
    }


def evaluate_model(model, dataloader, config, device, save_preds_dir=None):
    """Evaluate model on dataset.
    
    Args:
        model: trained model
        dataloader: test dataloader
        config: resolved config dict
        device: torch device
        save_preds_dir: if set, save prediction masks as PNG to this directory
    """
    model.eval()
    
    threshold = config.get('inference', {}).get('threshold', 0.5)
    keep_largest = config.get('postprocess', {}).get('keep_largest_cc', False)
    
    if save_preds_dir:
        save_preds_dir = Path(save_preds_dir)
        save_preds_dir.mkdir(parents=True, exist_ok=True)
    
    case_results = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            case_ids = batch['case_id']
            datasets = batch['dataset']
            
            # Forward
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            
            # Process each sample
            for i in range(len(case_ids)):
                pred = preds[i]
                mask = masks[i]
                
                # Optional post-processing
                if keep_largest:
                    pred_np = (pred.cpu().numpy() > threshold).astype(np.float32)
                    pred_np = keep_largest_cc(pred_np[0])[np.newaxis, ...]
                    pred = torch.from_numpy(pred_np).to(device)
                    metrics = compute_metrics(pred + 0.6, mask, threshold)
                else:
                    metrics = compute_metrics(pred, mask, threshold)
                
                # Save prediction mask as PNG
                if save_preds_dir:
                    pred_binary = (pred.cpu().squeeze().numpy() > threshold).astype(np.uint8) * 255
                    from PIL import Image as PILImage
                    PILImage.fromarray(pred_binary).save(
                        save_preds_dir / f'{case_ids[i]}_pred.png'
                    )
                
                case_results.append({
                    'case_id': case_ids[i],
                    'dataset': datasets[i],
                    **metrics
                })
    
    return case_results


def aggregate_results(case_results):
    """Aggregate per-case results to mean/std."""
    df = pd.DataFrame(case_results)
    
    agg = {}
    for metric in ['dice', 'iou', 'precision', 'recall']:
        agg[f'{metric}_mean'] = df[metric].mean()
        agg[f'{metric}_std'] = df[metric].std()
    
    agg['n_cases'] = len(df)
    
    return agg


def save_results(case_results, agg_results, output_dir, run_meta, args):
    """Save results in standardized format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Per-case results
    df_cases = pd.DataFrame(case_results)
    df_cases['dataset_train'] = run_meta.get('dataset_train', 'unknown')
    df_cases['model'] = run_meta.get('model', 'unknown')
    df_cases['seed'] = run_meta.get('seed', 42)
    df_cases['run_id'] = run_meta.get('run_id', 'unknown')
    df_cases['checkpoint'] = args.checkpoint_type
    df_cases['threshold'] = args.threshold
    
    cases_path = output_dir / 'results_cases.csv'
    df_cases.to_csv(cases_path, index=False)
    print(f"Saved per-case results to {cases_path}")
    
    # Aggregated results (long format)
    rows = []
    for metric in ['dice', 'iou', 'precision', 'recall']:
        rows.append({
            'dataset_train': run_meta.get('dataset_train', 'unknown'),
            'dataset_test': args.dataset_test,
            'model': run_meta.get('model', 'unknown'),
            'split': 'test',
            'checkpoint': args.checkpoint_type,
            'metric': metric,
            'value': agg_results[f'{metric}_mean'],
            'std': agg_results[f'{metric}_std'],
            'seed': run_meta.get('seed', 42),
            'run_id': run_meta.get('run_id', 'unknown'),
            'threshold': args.threshold,
            'n_cases': agg_results['n_cases'],
        })
    
    df_agg = pd.DataFrame(rows)
    agg_path = output_dir / 'results.csv'
    df_agg.to_csv(agg_path, index=False)
    print(f"Saved aggregated results to {agg_path}")
    
    return df_cases, df_agg


def main():
    parser = argparse.ArgumentParser(description='Evaluate segmentation model')
    
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--dataset', type=str, help='Dataset to evaluate on (default: same as training)')
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    parser.add_argument('--threshold', type=float, default=0.5, help='Binarization threshold')
    parser.add_argument('--keep-largest-cc', action='store_true', help='Keep largest connected component')
    parser.add_argument('--data-root', type=str, default=str(project_root))
    parser.add_argument('--output-dir', type=str, help='Output directory (default: checkpoint dir)')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--save-predictions', action='store_true',
                        help='Save prediction masks as PNG')
    
    args = parser.parse_args()
    
    # Determine checkpoint type
    ckpt_name = Path(args.checkpoint).stem
    args.checkpoint_type = 'best' if 'best' in ckpt_name else 'last'
    
    # Load checkpoint and config
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint, config = load_checkpoint(args.checkpoint)
    run_meta = get_run_meta(args.checkpoint)
    
    # Override config with CLI args
    if args.keep_largest_cc:
        config['postprocess']['keep_largest_cc'] = True
    config['inference']['threshold'] = args.threshold
    
    # Determine test dataset
    if args.dataset:
        dataset_filter = args.dataset if args.dataset != 'all' else None
        args.dataset_test = args.dataset
    else:
        dataset_filter = config['data'].get('dataset_filter')
        args.dataset_test = config['data'].get('dataset_name', 'all')
    
    print(f"Evaluating on: {args.dataset_test} ({args.split})")
    
    # Create model and load weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model: {config['model']['name']}")
    print(f"Device: {device}")
    
    # Create dataset
    dataset = ChestXrayDataset(
        data_root=args.data_root,
        index_path=config['data'].get('index_path', 'data/index.csv'),
        split=args.split,
        dataset=dataset_filter,
        image_size=config['dataset']['image_size']
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Determine prediction save directory
    save_preds_dir = None
    if args.save_predictions:
        if args.output_dir:
            save_preds_dir = Path(args.output_dir) / 'predictions'
        else:
            ckpt_dir = Path(args.checkpoint).parent.parent
            save_preds_dir = ckpt_dir / 'predictions' / args.dataset_test
    
    # Evaluate
    print("\nEvaluating...")
    case_results = evaluate_model(model, dataloader, config, device, save_preds_dir)
    
    # Aggregate
    agg_results = aggregate_results(case_results)
    
    print("\nResults:")
    print(f"  Dice: {agg_results['dice_mean']:.4f} +/- {agg_results['dice_std']:.4f}")
    print(f"  IoU:  {agg_results['iou_mean']:.4f} +/- {agg_results['iou_std']:.4f}")
    print(f"  Precision: {agg_results['precision_mean']:.4f}")
    print(f"  Recall: {agg_results['recall_mean']:.4f}")
    
    # Save results
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Save to checkpoint directory / results / {dataset_test}
        ckpt_dir = Path(args.checkpoint).parent.parent
        output_dir = ckpt_dir / 'results' / args.dataset_test
    
    save_results(case_results, agg_results, output_dir, run_meta, args)


if __name__ == '__main__':
    main()
