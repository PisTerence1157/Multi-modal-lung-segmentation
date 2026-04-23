#!/usr/bin/env python3
"""
Visualization Script

Generates comparison figures:
- Top-5, Bottom-5, Random-5 cases by Dice score
- Each figure shows: Original / GT / Model predictions

Usage:
    python scripts/visualize.py --results-cases outputs/.../results/kaggle/results_cases.csv \
                                --checkpoint outputs/.../checkpoints/best.pt
"""

import os
import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from models import get_model
from datasets.chest_xray_dataset import ChestXrayDataset


def load_checkpoint(checkpoint_path):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint, checkpoint['config']


def select_cases(results_df, n_top=5, n_bottom=5, n_random=5, seed=42):
    """Select top, bottom, and random cases."""
    df = results_df.copy()
    
    # Sort by dice
    df_sorted = df.sort_values('dice', ascending=False)
    
    # Top cases
    top_cases = df_sorted.head(n_top)['case_id'].tolist()
    
    # Bottom cases
    bottom_cases = df_sorted.tail(n_bottom)['case_id'].tolist()
    
    # Random cases (excluding top and bottom)
    remaining = df_sorted.iloc[n_top:-n_bottom] if len(df_sorted) > n_top + n_bottom else df_sorted
    if len(remaining) >= n_random:
        random_cases = remaining.sample(n=n_random, random_state=seed)['case_id'].tolist()
    else:
        random_cases = remaining['case_id'].tolist()
    
    return {
        'top': top_cases,
        'bottom': bottom_cases,
        'random': random_cases
    }


def get_case_dice(results_df, case_id):
    """Get dice score for a case."""
    row = results_df[results_df['case_id'] == case_id]
    if len(row) > 0:
        return row.iloc[0]['dice']
    return 0.0


def visualize_case(image, mask_gt, mask_pred, case_id, dice_score, output_path, threshold=0.5):
    """Generate visualization for a single case."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Convert tensors to numpy
    if torch.is_tensor(image):
        image = image.squeeze().cpu().numpy()
    if torch.is_tensor(mask_gt):
        mask_gt = mask_gt.squeeze().cpu().numpy()
    if torch.is_tensor(mask_pred):
        mask_pred = mask_pred.squeeze().cpu().numpy()
    
    # Denormalize image for display
    image_display = (image - image.min()) / (image.max() - image.min() + 1e-8)
    
    # Binarize prediction
    mask_pred_binary = (mask_pred > threshold).astype(np.float32)
    
    # Original image
    axes[0].imshow(image_display, cmap='gray')
    axes[0].set_title(f'Image: {case_id}')
    axes[0].axis('off')
    
    # Ground truth
    axes[1].imshow(mask_gt, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Prediction
    axes[2].imshow(mask_pred_binary, cmap='gray')
    axes[2].set_title(f'Prediction (Dice={dice_score:.4f})')
    axes[2].axis('off')
    
    # Overlay
    overlay = np.stack([image_display, image_display, image_display], axis=-1)
    
    # GT in green, pred in red, overlap in yellow
    gt_mask = mask_gt > 0.5
    pred_mask = mask_pred_binary > 0.5
    
    overlay_display = overlay.copy()
    overlay_display[gt_mask & ~pred_mask] = [0, 1, 0]  # Green: GT only (false negative)
    overlay_display[~gt_mask & pred_mask] = [1, 0, 0]  # Red: Pred only (false positive)
    overlay_display[gt_mask & pred_mask] = [1, 1, 0]   # Yellow: overlap (true positive)
    
    axes[3].imshow(overlay_display)
    axes[3].set_title('Overlay (G=GT, R=Pred, Y=Overlap)')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate visualizations')
    
    parser.add_argument('--results-cases', type=str, required=True,
                        help='Path to results_cases.csv')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--n-top', type=int, default=5, help='Number of top cases')
    parser.add_argument('--n-bottom', type=int, default=5, help='Number of bottom cases')
    parser.add_argument('--n-random', type=int, default=5, help='Number of random cases')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--data-root', type=str, default=str(project_root))
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from: {args.results_cases}")
    results_df = pd.read_csv(args.results_cases)
    
    # Select cases
    selected = select_cases(results_df, args.n_top, args.n_bottom, args.n_random)
    all_case_ids = selected['top'] + selected['bottom'] + selected['random']
    
    print(f"Selected cases:")
    print(f"  Top {args.n_top}: {selected['top']}")
    print(f"  Bottom {args.n_bottom}: {selected['bottom']}")
    print(f"  Random {args.n_random}: {selected['random']}")
    
    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint, config = load_checkpoint(args.checkpoint)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Determine dataset
    dataset_filter = results_df['dataset'].iloc[0] if 'dataset' in results_df.columns else None
    dataset_test = results_df['dataset_train'].iloc[0] if 'dataset_train' in results_df.columns else 'unknown'
    
    # Create dataset
    dataset = ChestXrayDataset(
        data_root=args.data_root,
        index_path=config['data'].get('index_path', 'data/index.csv'),
        split='test',
        dataset=dataset_filter,
        image_size=config['dataset']['image_size']
    )
    
    # Build case_id to index mapping
    case_to_idx = {}
    for idx in range(len(dataset)):
        info = dataset.get_sample_info(idx)
        case_to_idx[info['case_id']] = idx
    
    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ckpt_dir = Path(args.checkpoint).parent.parent
        output_dir = ckpt_dir / 'visualizations' / dataset_test
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Generate visualizations
    for category, case_ids in selected.items():
        cat_dir = output_dir / category
        cat_dir.mkdir(exist_ok=True)
        
        for case_id in case_ids:
            if case_id not in case_to_idx:
                print(f"Warning: case {case_id} not found in dataset")
                continue
            
            idx = case_to_idx[case_id]
            sample = dataset[idx]
            
            image = sample['image'].unsqueeze(0).to(device)
            mask_gt = sample['mask']
            
            # Predict
            with torch.no_grad():
                output = model(image)
                mask_pred = torch.sigmoid(output).cpu()
            
            # Get dice score
            dice = get_case_dice(results_df, case_id)
            
            # Save visualization
            output_path = cat_dir / f'{case_id}_dice_{dice:.3f}.png'
            visualize_case(
                sample['image'], mask_gt, mask_pred.squeeze(0),
                case_id, dice, output_path, args.threshold
            )
            print(f"  Saved: {output_path.name}")
    
    print(f"\nVisualization complete! {len(all_case_ids)} images saved to {output_dir}")


if __name__ == '__main__':
    main()
