#!/usr/bin/env python3
"""
LUNA16 Dataset Preprocessing Script

Converts LUNA16 .mhd/.raw format to .nii.gz for compatibility with existing pipeline.
Generates index_luna16.csv with case-level split.

Usage:
    python scripts/preprocess_luna16.py --data-root /path/to/data
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

# SimpleITK for reading .mhd/.raw
try:
    import SimpleITK as sitk
except ImportError:
    print("Error: SimpleITK not installed. Run: pip install SimpleITK")
    sys.exit(1)

# nibabel for saving .nii.gz
import nibabel as nib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Configuration
LUNA16_CT_DIRS = [
    "data/new data/CT NEW/subset0",
    "data/new data/CT NEW/subset1",
    "data/new data/CT NEW/subset2",
    "data/new data/CT NEW/subset3",
]
LUNA16_MASK_DIR = "data/new data/CT NEW/seg-lungs-LUNA16"
OUTPUT_DIR = "data/luna16_processed"
INDEX_OUTPUT = "data/index_luna16.csv"
STATS_OUTPUT = "data/stats_luna16.json"
PREVIEW_DIR = "data/preview_luna16"

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42


def load_mhd_volume(mhd_path: Path) -> Tuple[np.ndarray, np.ndarray, Tuple]:
    """
    Load .mhd/.raw volume using SimpleITK.
    
    Returns:
        data: numpy array (D, H, W)
        spacing: voxel spacing (x, y, z)
        origin: volume origin
    """
    img = sitk.ReadImage(str(mhd_path))
    data = sitk.GetArrayFromImage(img)  # (D, H, W)
    spacing = img.GetSpacing()  # (x, y, z)
    origin = img.GetOrigin()
    direction = img.GetDirection()
    
    return data, np.array(spacing), origin, direction


def save_as_nifti(data: np.ndarray, spacing: np.ndarray, output_path: Path):
    """
    Save numpy array as NIfTI .nii.gz file.
    
    Args:
        data: numpy array (D, H, W)
        spacing: voxel spacing (x, y, z)
        output_path: output file path
    """
    # Create affine matrix from spacing
    # NIfTI uses RAS+ orientation, spacing is (x, y, z)
    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
    
    # nibabel expects (H, W, D) or (X, Y, Z), but we have (D, H, W)
    # Transpose to (H, W, D)
    data_transposed = np.transpose(data, (1, 2, 0))
    
    nii_img = nib.Nifti1Image(data_transposed, affine)
    nib.save(nii_img, str(output_path))


def find_matching_mask(ct_uid: str, mask_dir: Path) -> Path:
    """Find matching mask file for a CT scan."""
    # LUNA16 naming: CT is like 1.3.6.1.4.1.14519.5.2.1...
    # Mask has same UID
    mask_path = mask_dir / f"{ct_uid}.mhd"
    if mask_path.exists():
        return mask_path
    return None


def create_preview(ct_data: np.ndarray, mask_data: np.ndarray, 
                   case_id: str, output_dir: Path):
    """Create preview visualization."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Select middle slice
    mid_slice = ct_data.shape[0] // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # CT slice
    axes[0].imshow(ct_data[mid_slice], cmap='gray')
    axes[0].set_title(f'CT (slice {mid_slice})')
    axes[0].axis('off')
    
    # Mask slice
    axes[1].imshow(mask_data[mid_slice], cmap='gray')
    axes[1].set_title('Lung Mask')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(ct_data[mid_slice], cmap='gray')
    axes[2].imshow(mask_data[mid_slice], cmap='Reds', alpha=0.3)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.suptitle(f'Case: {case_id[:20]}...')
    plt.tight_layout()
    plt.savefig(output_dir / f'{case_id[:30]}.png', dpi=100, bbox_inches='tight')
    plt.close()


def process_luna16(data_root: Path, dry_run: bool = False):
    """Process LUNA16 dataset."""
    
    output_dir = data_root / OUTPUT_DIR
    preview_dir = data_root / PREVIEW_DIR
    mask_dir = data_root / LUNA16_MASK_DIR
    
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        preview_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all CT scans
    ct_files = []
    for subset_dir in LUNA16_CT_DIRS:
        subset_path = data_root / subset_dir
        if subset_path.exists():
            mhd_files = list(subset_path.glob("*.mhd"))
            ct_files.extend(mhd_files)
            print(f"Found {len(mhd_files)} CT scans in {subset_dir}")
    
    print(f"\nTotal CT scans found: {len(ct_files)}")
    
    # Process each CT scan
    records = []
    stats = {
        'total_cases': 0,
        'processed_cases': 0,
        'skipped_no_mask': 0,
        'depths': [],
        'mask_ratios': []
    }
    
    for i, ct_path in enumerate(ct_files):
        case_id = ct_path.stem
        print(f"\n[{i+1}/{len(ct_files)}] Processing {case_id[:30]}...")
        
        # Find matching mask
        mask_path = find_matching_mask(case_id, mask_dir)
        if mask_path is None:
            print(f"  Warning: No mask found, skipping")
            stats['skipped_no_mask'] += 1
            continue
        
        stats['total_cases'] += 1
        
        if dry_run:
            print(f"  [DRY RUN] Would process CT and mask")
            continue
        
        try:
            # Load CT
            ct_data, ct_spacing, ct_origin, ct_direction = load_mhd_volume(ct_path)
            print(f"  CT shape: {ct_data.shape}, spacing: {ct_spacing}")
            
            # Load mask
            mask_data, mask_spacing, _, _ = load_mhd_volume(mask_path)
            print(f"  Mask shape: {mask_data.shape}")
            
            # Verify shapes match
            if ct_data.shape != mask_data.shape:
                print(f"  Warning: Shape mismatch CT{ct_data.shape} vs Mask{mask_data.shape}, skipping")
                continue
            
            # Binarize mask (LUNA16 masks should already be binary)
            mask_binary = (mask_data > 0).astype(np.float32)
            mask_ratio = mask_binary.sum() / mask_binary.size
            
            if mask_ratio < 0.01:
                print(f"  Warning: Very low mask ratio ({mask_ratio:.4f})")
            
            # Save as NIfTI
            ct_output = output_dir / f"{case_id}_image.nii.gz"
            mask_output = output_dir / f"{case_id}_mask.nii.gz"
            
            save_as_nifti(ct_data, ct_spacing, ct_output)
            save_as_nifti(mask_binary, mask_spacing, mask_output)
            
            print(f"  Saved: {ct_output.name}, {mask_output.name}")
            
            # Record
            records.append({
                'dataset': 'luna16',
                'modality': 'CT',
                'source': 'luna16',
                'case_id': case_id,
                'image_path': str(ct_output.relative_to(data_root)),
                'mask_path': str(mask_output.relative_to(data_root)),
                'split': '',  # Will be assigned later
                'depth': ct_data.shape[0],
                'height': ct_data.shape[1],
                'width': ct_data.shape[2],
                'spacing_x': float(ct_spacing[0]),
                'spacing_y': float(ct_spacing[1]),
                'spacing_z': float(ct_spacing[2]),
                'mask_volume_ratio': float(mask_ratio)
            })
            
            stats['processed_cases'] += 1
            stats['depths'].append(ct_data.shape[0])
            stats['mask_ratios'].append(mask_ratio)
            
            # Create preview for first 20 cases
            if len(records) <= 20:
                create_preview(ct_data, mask_binary, case_id, preview_dir)
                
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    if dry_run:
        print(f"\n[DRY RUN] Would process {stats['total_cases']} cases")
        return
    
    if len(records) == 0:
        print("No records to save!")
        return
    
    # Assign splits (case-level)
    random.seed(SEED)
    random.shuffle(records)
    
    n_total = len(records)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    
    for i, record in enumerate(records):
        if i < n_train:
            record['split'] = 'train'
        elif i < n_train + n_val:
            record['split'] = 'val'
        else:
            record['split'] = 'test'
    
    # Save index
    df = pd.DataFrame(records)
    index_path = data_root / INDEX_OUTPUT
    df.to_csv(index_path, index=False)
    print(f"\nSaved index to {index_path}")
    print(f"  Train: {len(df[df['split']=='train'])}")
    print(f"  Val: {len(df[df['split']=='val'])}")
    print(f"  Test: {len(df[df['split']=='test'])}")
    
    # Save stats
    stats_summary = {
        'total_cases': stats['total_cases'],
        'processed_cases': stats['processed_cases'],
        'skipped_no_mask': stats['skipped_no_mask'],
        'train_count': len(df[df['split']=='train']),
        'val_count': len(df[df['split']=='val']),
        'test_count': len(df[df['split']=='test']),
        'depth_mean': float(np.mean(stats['depths'])),
        'depth_std': float(np.std(stats['depths'])),
        'mask_ratio_mean': float(np.mean(stats['mask_ratios'])),
        'mask_ratio_std': float(np.std(stats['mask_ratios'])),
    }
    
    stats_path = data_root / STATS_OUTPUT
    with open(stats_path, 'w') as f:
        json.dump(stats_summary, f, indent=2)
    print(f"Saved stats to {stats_path}")
    
    print(f"\n=== Summary ===")
    print(f"Processed: {stats['processed_cases']} cases")
    print(f"Skipped (no mask): {stats['skipped_no_mask']}")
    print(f"Average depth: {np.mean(stats['depths']):.1f} slices")
    print(f"Average mask ratio: {np.mean(stats['mask_ratios']):.3f}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess LUNA16 dataset')
    parser.add_argument('--data-root', type=str, 
                        default=str(Path(__file__).parent.parent),
                        help='Data root directory')
    parser.add_argument('--dry-run', action='store_true',
                        help='Dry run (no file writes)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("LUNA16 Preprocessing")
    print("="*60)
    print(f"Data root: {args.data_root}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Index: {INDEX_OUTPUT}")
    
    process_luna16(Path(args.data_root), args.dry_run)


if __name__ == '__main__':
    main()
