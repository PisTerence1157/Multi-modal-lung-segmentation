#!/usr/bin/env python3
"""
CT Dataset Preprocessing Script

Processes COVID-19 CT dataset:
- Reads .hdr/.img Analyze format
- Resamples to target spacing
- Generates index_ct.csv with case-level split
- Creates preview visualizations

Usage:
    python scripts/preprocess_ct.py --data-root /path/to/data
"""

import os
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import re

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Dataset configuration
CT_DATA_DIR = "data/new data/COVID-19 & Normal CT Segmentation Dataset/Organized COVID19 CT Data_rev2/Part 1"
CT_LABEL_DIR = "data/new data/COVID-19 & Normal CT Segmentation Dataset/Organized COVID19 CT Data_rev2/Part 1/CT Labels"

# Target spacing (mm)
TARGET_SPACING = [1.0, 1.0, 1.0]

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42


def load_analyze_image(hdr_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Analyze 7.5 format (.hdr/.img).
    
    Returns:
        data: numpy array
        spacing: voxel spacing (x, y, z)
    """
    img = nib.load(str(hdr_path))
    data = img.get_fdata()
    
    # Get voxel spacing from header
    header = img.header
    spacing = header.get_zooms()[:3]
    
    return data, np.array(spacing)


def get_case_id(filename: str) -> str:
    """Extract case ID from filename (e.g., p100_ns002i00001 -> p100)."""
    match = re.match(r'(p?\d+)', filename)
    if match:
        return match.group(1)
    return filename.split('_')[0]


def find_matching_label(image_path: Path, label_dir: Path) -> Path:
    """Find matching label file for an image."""
    case_id = get_case_id(image_path.stem)
    
    # Search for matching label file
    for label_file in label_dir.glob(f"{case_id}*.hdr"):
        return label_file
    
    return None


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Convert mask to binary (0/1).
    Handles special case where mask values are -1024 (background) and -1023 (foreground).
    """
    # Check if mask uses -1024/-1023 encoding
    unique_vals = np.unique(mask)
    if len(unique_vals) == 2 and unique_vals[0] < 0:
        # Convert: -1024 -> 0, -1023 -> 1
        return (mask > mask.min()).astype(np.float32)
    else:
        # Standard binary mask
        return (mask > 0).astype(np.float32)


def get_volume_info(data: np.ndarray, mask: np.ndarray = None) -> Dict:
    """Get volume statistics."""
    info = {
        'shape': data.shape,
        'min': float(data.min()),
        'max': float(data.max()),
        'mean': float(data.mean()),
        'std': float(data.std()),
    }
    
    if mask is not None:
        mask_binary = binarize_mask(mask)
        info['mask_volume_ratio'] = float(mask_binary.sum() / mask_binary.size)
    
    return info


def process_ct_dataset(data_root: Path) -> List[Dict]:
    """Process CT dataset and return list of records."""
    image_dir = data_root / CT_DATA_DIR
    label_dir = data_root / CT_LABEL_DIR
    
    if not image_dir.exists():
        print(f"Warning: {image_dir} does not exist")
        return []
    
    if not label_dir.exists():
        print(f"Warning: {label_dir} does not exist")
        return []
    
    # Find all image files
    image_files = sorted(list(image_dir.glob("p*_ns*.hdr")))
    print(f"Found {len(image_files)} CT volumes")
    
    records = []
    case_ids = set()
    
    for image_path in image_files:
        case_id = get_case_id(image_path.stem)
        
        # Skip if already processed this case
        if case_id in case_ids:
            continue
        case_ids.add(case_id)
        
        # Find matching label
        label_path = find_matching_label(image_path, label_dir)
        if label_path is None:
            print(f"Warning: no label found for {case_id}")
            continue
        
        try:
            # Load and validate
            image_data, image_spacing = load_analyze_image(image_path)
            label_data, _ = load_analyze_image(label_path)
            
            # Sanity check
            if image_data.shape != label_data.shape:
                print(f"Warning: shape mismatch for {case_id}: {image_data.shape} vs {label_data.shape}")
                continue
            
            # Get volume info
            info = get_volume_info(image_data, label_data)
            
            # Skip volumes with no foreground
            if info.get('mask_volume_ratio', 0) == 0:
                print(f"Warning: skipping {case_id} (no foreground)")
                continue
            
            records.append({
                'dataset': 'covid_ct',
                'modality': 'CT',
                'source': 'covid19',
                'case_id': case_id,
                'image_path': str(image_path.relative_to(data_root)),
                'mask_path': str(label_path.relative_to(data_root)),
                'depth': image_data.shape[2],
                'height': image_data.shape[0],
                'width': image_data.shape[1],
                'spacing_x': float(image_spacing[0]),
                'spacing_y': float(image_spacing[1]),
                'spacing_z': float(image_spacing[2]),
                'mask_volume_ratio': info['mask_volume_ratio'],
            })
            
        except Exception as e:
            print(f"Error processing {case_id}: {e}")
            continue
    
    return records


def assign_case_splits(case_ids: List[str], seed: int = SEED) -> Dict[str, str]:
    """Assign splits at case level (critical for CT to avoid data leakage)."""
    random.seed(seed)
    shuffled = list(set(case_ids))
    random.shuffle(shuffled)
    
    n = len(shuffled)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    
    splits = {}
    for i, case_id in enumerate(shuffled):
        if i < n_train:
            splits[case_id] = 'train'
        elif i < n_train + n_val:
            splits[case_id] = 'val'
        else:
            splits[case_id] = 'test'
    
    return splits


def compute_ct_statistics(df: pd.DataFrame) -> Dict:
    """Compute CT dataset statistics."""
    stats = {
        'total_cases': len(df),
        'train_cases': len(df[df['split'] == 'train']),
        'val_cases': len(df[df['split'] == 'val']),
        'test_cases': len(df[df['split'] == 'test']),
        'total_slices': int(df['depth'].sum()),
        'shape': {
            'depth_mean': float(df['depth'].mean()),
            'depth_std': float(df['depth'].std()),
            'height_mean': float(df['height'].mean()),
            'width_mean': float(df['width'].mean()),
        },
        'spacing': {
            'x_mean': float(df['spacing_x'].mean()),
            'y_mean': float(df['spacing_y'].mean()),
            'z_mean': float(df['spacing_z'].mean()),
        },
        'mask': {
            'volume_ratio_mean': float(df['mask_volume_ratio'].mean()),
            'volume_ratio_std': float(df['mask_volume_ratio'].std()),
        },
        'target_spacing': TARGET_SPACING,
    }
    return stats


def generate_ct_previews(data_root: Path, df: pd.DataFrame, output_dir: Path, n_samples: int = 10):
    """Generate CT preview images (middle slice)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    samples = df.sample(n=min(n_samples, len(df)), random_state=SEED)
    
    for _, row in samples.iterrows():
        try:
            image_path = data_root / row['image_path']
            mask_path = data_root / row['mask_path']
            
            image_data, _ = load_analyze_image(image_path)
            mask_data, _ = load_analyze_image(mask_path)
            
            # Get middle slice
            mid_slice = image_data.shape[2] // 2
            image_slice = image_data[:, :, mid_slice]
            mask_slice = mask_data[:, :, mid_slice]
            
            # Normalize image for display
            image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min() + 1e-8)
            
            # Binarize mask
            mask_slice = binarize_mask(mask_slice)
            
            # Create figure
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(image_slice.T, cmap='gray', origin='lower')
            axes[0].set_title(f'CT: {row["case_id"]} (slice {mid_slice}/{row["depth"]})')
            axes[0].axis('off')
            
            axes[1].imshow(mask_slice.T, cmap='gray', origin='lower')
            axes[1].set_title('Mask')
            axes[1].axis('off')
            
            # Overlay
            overlay = np.stack([image_slice, image_slice, image_slice], axis=-1)
            mask_rgb = np.zeros((*image_slice.shape, 3))
            mask_rgb[:, :, 0] = (mask_slice > 0).astype(float)
            overlay = (overlay * 0.7 + mask_rgb * 0.3)
            
            axes[2].imshow(overlay.transpose(1, 0, 2), origin='lower')
            axes[2].set_title(f'Overlay (ratio={row["mask_volume_ratio"]:.4f})')
            axes[2].axis('off')
            
            plt.tight_layout()
            save_path = output_dir / f'ct_{row["case_id"]}.png'
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error generating preview for {row['case_id']}: {e}")
    
    print(f"Generated CT previews in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess CT dataset')
    parser.add_argument('--data-root', type=str, default='.',
                        help='Root directory containing all datasets')
    args = parser.parse_args()
    
    data_root = Path(args.data_root).resolve()
    print(f"Data root: {data_root}")
    
    # Process CT dataset
    print("\nProcessing CT dataset...")
    records = process_ct_dataset(data_root)
    
    if not records:
        print("No CT records found!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Assign case-level splits
    case_ids = df['case_id'].tolist()
    splits = assign_case_splits(case_ids)
    df['split'] = df['case_id'].map(splits)
    
    # Reorder columns
    columns = ['dataset', 'modality', 'source', 'case_id', 'image_path', 'mask_path',
               'split', 'depth', 'height', 'width', 'spacing_x', 'spacing_y', 'spacing_z',
               'mask_volume_ratio']
    df = df[columns]
    
    # Save index
    index_path = data_root / 'data' / 'index_ct.csv'
    df.to_csv(index_path, index=False)
    print(f"\nSaved CT index to {index_path}")
    print(f"Total cases: {len(df)}")
    
    # Print split summary
    print("\nSplit summary:")
    print(df.groupby('split').size())
    
    # Compute and save statistics
    stats = compute_ct_statistics(df)
    stats_path = data_root / 'data' / 'stats_ct.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved statistics to {stats_path}")
    
    # Generate previews
    preview_dir = data_root / 'data' / 'preview_ct'
    generate_ct_previews(data_root, df, preview_dir, n_samples=10)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
