#!/usr/bin/env python3
"""
CT Evaluation Script

Evaluates CT segmentation models with volume-level metrics.

For 2D models: aggregates slice predictions into volume, then computes volume Dice.
For 3D models: directly computes volume Dice.

Outputs:
- results_cases.csv: per-case metrics
- results.csv: aggregated metrics (mean/std)

Usage:
    python scripts/evaluate_ct.py --checkpoint outputs/luna16_2d/unet/.../checkpoints/best.pt
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
import yaml
import nibabel as nib

from models import get_model, UNet3D


def load_checkpoint(checkpoint_path):
    """Load model checkpoint and config."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    mode = checkpoint.get('mode', '2d')
    return checkpoint, config, mode


def get_run_meta(checkpoint_path):
    """Load run metadata from checkpoint directory."""
    ckpt_dir = Path(checkpoint_path).parent
    run_dir = ckpt_dir.parent
    
    meta_path = run_dir / 'run_meta.json'
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            return json.load(f)
    return {}


def compute_volume_metrics(pred_volume, target_volume, threshold=0.5):
    """
    Compute volume-level segmentation metrics.
    
    Args:
        pred_volume: predicted volume (D, H, W) - probabilities
        target_volume: ground truth volume (D, H, W) - binary
        threshold: binarization threshold
    
    Returns:
        dict with dice, iou, precision, recall, accuracy
    """
    # Binarize prediction
    pred_binary = (pred_volume > threshold).astype(np.float32)
    target_binary = target_volume.astype(np.float32)
    
    # Flatten
    pred_flat = pred_binary.flatten()
    target_flat = target_binary.flatten()
    
    # TP, FP, FN, TN
    tp = np.sum(pred_flat * target_flat)
    fp = np.sum(pred_flat * (1 - target_flat))
    fn = np.sum((1 - pred_flat) * target_flat)
    tn = np.sum((1 - pred_flat) * (1 - target_flat))
    
    # Metrics
    eps = 1e-7
    
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    
    return {
        'dice': float(dice),
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall),
        'accuracy': float(accuracy)
    }


def load_nifti_volume(path):
    """Load NIfTI volume."""
    img = nib.load(str(path))
    data = img.get_fdata().astype(np.float32)
    # nibabel loads as (H, W, D), transpose to (D, H, W)
    data = np.transpose(data, (2, 0, 1))
    return data


def normalize_ct(volume, window_center=-600, window_width=1500):
    """Normalize CT volume using windowing."""
    min_val = window_center - window_width / 2
    max_val = window_center + window_width / 2
    volume = np.clip(volume, min_val, max_val)
    volume = (volume - min_val) / (max_val - min_val)
    return volume


def evaluate_2d_model(model, index_df, data_root, config, device, threshold=0.5,
                      save_preds_dir=None):
    """
    Evaluate 2D model on CT volumes.
    
    Process each volume slice-by-slice, aggregate predictions, compute volume Dice.
    """
    results = []
    
    # Get unique cases
    cases = index_df[index_df['split'] == 'test'].to_dict('records')
    
    ct_cfg = config.get('ct', {})
    window_center = float(ct_cfg.get('window_center', -600))
    window_width = float(ct_cfg.get('window_width', 1500))
    image_size = config.get('dataset', {}).get('image_size', [512, 512])
    
    model.eval()
    
    with torch.no_grad():
        for case in tqdm(cases, desc='Evaluating cases'):
            case_id = case['case_id']
            
            # Load volume
            image_path = Path(data_root) / case['image_path']
            mask_path = Path(data_root) / case['mask_path']
            
            volume = load_nifti_volume(image_path)
            mask = load_nifti_volume(mask_path)
            mask = (mask > 0).astype(np.float32)
            
            # Normalize
            volume = normalize_ct(volume, window_center, window_width)
            
            # Predict slice-by-slice
            pred_volume = np.zeros_like(volume)
            
            for i in range(volume.shape[0]):
                slice_img = volume[i]
                
                # Resize if needed
                if slice_img.shape != tuple(image_size):
                    slice_img = F.interpolate(
                        torch.from_numpy(slice_img).unsqueeze(0).unsqueeze(0),
                        size=image_size, mode='bilinear', align_corners=False
                    ).squeeze().numpy()
                
                # Predict
                input_tensor = torch.from_numpy(slice_img).unsqueeze(0).unsqueeze(0).float().to(device)
                output = model(input_tensor)
                pred = torch.sigmoid(output).squeeze().cpu().numpy()
                
                # Resize back if needed
                if pred.shape != volume[i].shape:
                    pred = F.interpolate(
                        torch.from_numpy(pred).unsqueeze(0).unsqueeze(0),
                        size=volume[i].shape, mode='bilinear', align_corners=False
                    ).squeeze().numpy()
                
                pred_volume[i] = pred
            
            # Save prediction slices for visualization
            if save_preds_dir:
                case_dir = save_preds_dir / case_id
                case_dir.mkdir(parents=True, exist_ok=True)
                # Save 5 representative slices (evenly spaced through lung region)
                fg_slices = np.where(mask.sum(axis=(1, 2)) > 0)[0]
                if len(fg_slices) > 0:
                    indices = np.linspace(fg_slices[0], fg_slices[-1], 5, dtype=int)
                else:
                    indices = np.linspace(0, volume.shape[0] - 1, 5, dtype=int)
                for si in indices:
                    np.savez_compressed(
                        case_dir / f'slice_{si:04d}.npz',
                        image=volume[si], mask=mask[si],
                        pred=pred_volume[si]
                    )
            
            # Compute volume metrics
            metrics = compute_volume_metrics(pred_volume, mask, threshold)
            metrics['case_id'] = case_id
            results.append(metrics)
    
    return results


def evaluate_3d_model(model, index_df, data_root, config, device, threshold=0.5,
                      save_preds_dir=None):
    """
    Evaluate 3D model on CT volumes using sliding window inference.
    """
    results = []
    
    cases = index_df[index_df['split'] == 'test'].to_dict('records')
    
    ct_cfg = config.get('ct', {})
    window_center = float(ct_cfg.get('window_center', -600))
    window_width = float(ct_cfg.get('window_width', 1500))
    patch_size = ct_cfg.get('patch_size', [128, 128, 32])
    
    model.eval()
    
    with torch.no_grad():
        for case in tqdm(cases, desc='Evaluating cases'):
            case_id = case['case_id']
            
            # Load volume
            image_path = Path(data_root) / case['image_path']
            mask_path = Path(data_root) / case['mask_path']
            
            volume = load_nifti_volume(image_path)
            mask = load_nifti_volume(mask_path)
            mask = (mask > 0).astype(np.float32)
            
            # Normalize
            volume = normalize_ct(volume, window_center, window_width)
            
            # Sliding window inference
            pred_volume = sliding_window_inference(
                model, volume, patch_size, device, overlap=0.5
            )
            
            # Save prediction slices for visualization
            if save_preds_dir:
                case_dir = save_preds_dir / case_id
                case_dir.mkdir(parents=True, exist_ok=True)
                fg_slices = np.where(mask.sum(axis=(1, 2)) > 0)[0]
                if len(fg_slices) > 0:
                    indices = np.linspace(fg_slices[0], fg_slices[-1], 5, dtype=int)
                else:
                    indices = np.linspace(0, volume.shape[0] - 1, 5, dtype=int)
                for si in indices:
                    np.savez_compressed(
                        case_dir / f'slice_{si:04d}.npz',
                        image=volume[si], mask=mask[si],
                        pred=pred_volume[si]
                    )
            
            # Compute volume metrics
            metrics = compute_volume_metrics(pred_volume, mask, threshold)
            metrics['case_id'] = case_id
            results.append(metrics)
    
    return results


def sliding_window_inference(model, volume, patch_size, device, overlap=0.5):
    """
    Perform sliding window inference on a 3D volume.
    Pads the volume if any dimension is smaller than patch_size.
    """
    orig_shape = volume.shape
    D, H, W = volume.shape
    pd, ph, pw = patch_size
    
    # Pad volume if smaller than patch_size in any dimension
    pad_d = max(0, pd - D)
    pad_h = max(0, ph - H)
    pad_w = max(0, pw - W)
    
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        volume = np.pad(volume,
                        ((0, pad_d), (0, pad_h), (0, pad_w)),
                        mode='constant', constant_values=0)
        D, H, W = volume.shape
    
    # Calculate stride
    stride_d = max(1, int(pd * (1 - overlap)))
    stride_h = max(1, int(ph * (1 - overlap)))
    stride_w = max(1, int(pw * (1 - overlap)))
    
    # Output volume and count (for averaging overlaps)
    pred_sum = np.zeros((D, H, W), dtype=np.float32)
    count = np.zeros((D, H, W), dtype=np.float32)
    
    # Sliding window
    for d in range(0, max(D - pd + 1, 1), stride_d):
        for h in range(0, max(H - ph + 1, 1), stride_h):
            for w in range(0, max(W - pw + 1, 1), stride_w):
                # Handle edge cases
                d_end = min(d + pd, D)
                h_end = min(h + ph, H)
                w_end = min(w + pw, W)
                d_start = d_end - pd
                h_start = h_end - ph
                w_start = w_end - pw
                
                # Extract patch
                patch = volume[d_start:d_end, h_start:h_end, w_start:w_end]
                
                # Predict
                input_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
                output = model(input_tensor)
                pred = torch.sigmoid(output).squeeze().cpu().numpy()
                
                # Accumulate
                pred_sum[d_start:d_end, h_start:h_end, w_start:w_end] += pred
                count[d_start:d_end, h_start:h_end, w_start:w_end] += 1
    
    # Average
    pred_volume = pred_sum / np.maximum(count, 1)
    
    # Crop back to original size
    pred_volume = pred_volume[:orig_shape[0], :orig_shape[1], :orig_shape[2]]
    
    return pred_volume


def main():
    parser = argparse.ArgumentParser(description='Evaluate CT segmentation model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-root', type=str, default=str(project_root),
                        help='Data root directory')
    parser.add_argument('--index-path', type=str, default=None,
                        help='Path to index CSV (default: from checkpoint config)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Binarization threshold')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: checkpoint run dir)')
    parser.add_argument('--save-predictions', action='store_true',
                        help='Save prediction slices as .npz for visualization')
    
    args = parser.parse_args()
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint, config, mode = load_checkpoint(args.checkpoint)
    
    # Get run metadata
    run_meta = get_run_meta(args.checkpoint)
    
    # Determine index path
    if args.index_path:
        index_path = args.index_path
    else:
        index_path = config.get('index_path', 'data/index_ct.csv')
    
    # Load index
    full_index_path = Path(args.data_root) / index_path
    print(f"Loading index: {full_index_path}")
    index_df = pd.read_csv(full_index_path)
    
    test_count = len(index_df[index_df['split'] == 'test'])
    print(f"Test cases: {test_count}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create model
    print(f"Creating model (mode={mode})...")
    if mode == '3d':
        model = UNet3D(
            in_channels=config['model'].get('in_channels', 1),
            out_channels=config['model'].get('out_channels', 1),
            features=config['model'].get('features', [32, 64, 128, 256]),
            dropout=config['model'].get('dropout', 0.0)
        )
    else:
        model = get_model(config)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Determine prediction save directory
    save_preds_dir = None
    if args.save_predictions:
        if args.output_dir:
            save_preds_dir = Path(args.output_dir) / 'predictions'
        else:
            save_preds_dir = Path(args.checkpoint).parent.parent / 'predictions'
    
    # Evaluate
    print(f"\nEvaluating...")
    if mode == '3d':
        results = evaluate_3d_model(model, index_df, args.data_root, config, device,
                                     args.threshold, save_preds_dir)
    else:
        results = evaluate_2d_model(model, index_df, args.data_root, config, device,
                                     args.threshold, save_preds_dir)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.checkpoint).parent.parent / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save per-case results
    cases_path = output_dir / 'results_cases.csv'
    results_df.to_csv(cases_path, index=False)
    print(f"\nSaved per-case results to {cases_path}")
    
    # Compute aggregated metrics
    metrics = ['dice', 'iou', 'precision', 'recall', 'accuracy']
    agg_results = []
    
    for metric in metrics:
        agg_results.append({
            'metric': metric,
            'mean': results_df[metric].mean(),
            'std': results_df[metric].std(),
            'min': results_df[metric].min(),
            'max': results_df[metric].max()
        })
    
    agg_df = pd.DataFrame(agg_results)
    
    # Add metadata
    dataset_name = config.get('dataset_name', 'ct')
    model_name = config['model']['name']
    seed = config['training'].get('seed', 42)
    
    # Save aggregated results
    summary_path = output_dir / 'results.csv'
    
    # Long format for consistency
    summary_records = []
    for _, row in agg_df.iterrows():
        summary_records.append({
            'dataset_train': run_meta.get('dataset_train', f'{dataset_name}_{mode}'),
            'dataset_test': f'{dataset_name}_{mode}',
            'model': model_name,
            'mode': mode,
            'split': 'test',
            'checkpoint': 'best',
            'metric': row['metric'],
            'value': row['mean'],
            'std': row['std'],
            'seed': seed,
            'threshold': args.threshold
        })
    
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved aggregated results to {summary_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Mode: {mode}")
    print(f"Dataset: {dataset_name}")
    print(f"Test cases: {len(results_df)}")
    print("-"*40)
    for _, row in agg_df.iterrows():
        print(f"{row['metric']:12s}: {row['mean']:.4f} +/- {row['std']:.4f}")
    print("="*60)


if __name__ == '__main__':
    main()
