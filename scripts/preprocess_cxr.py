#!/usr/bin/env python3
"""
CXR Dataset Preprocessing Script

Generates unified index.csv for all CXR datasets:
- Kaggle (existing)
- Montgomery (Mendeley)
- Shenzhen (Mendeley)

Outputs:
- data/index.csv: unified dataset index
- data/stats_cxr.json: dataset statistics
- data/preview/: 20 overlay visualizations per dataset

Usage:
    python scripts/preprocess_cxr.py --data-root /path/to/data
"""

import os
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt


# Dataset configurations
DATASETS = {
    'kaggle': {
        'modality': 'CXR',
        'source': 'kaggle',
        'image_dir': 'Chest-X-Ray/image',
        'mask_dir': 'Chest-X-Ray/mask',
    },
    'montgomery': {
        'modality': 'CXR',
        'source': 'mendeley',
        'image_dir': 'data/new data/Chest X-ray dataset for lung segmentation/Montgomery/img',
        'mask_dir': 'data/new data/Chest X-ray dataset for lung segmentation/Montgomery/mask',
    },
    'shenzhen': {
        'modality': 'CXR',
        'source': 'mendeley',
        'image_dir': 'data/new data/Chest X-ray dataset for lung segmentation/Shenzhen/img',
        'mask_dir': 'data/new data/Chest X-ray dataset for lung segmentation/Shenzhen/mask',
    },
}

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42


def get_image_info(image_path: Path, mask_path: Path) -> Dict:
    """Get image dimensions and mask statistics."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
        
        with Image.open(mask_path) as mask:
            mask_arr = np.array(mask)
            # Binarize mask
            if mask_arr.max() > 1:
                mask_arr = (mask_arr > 127).astype(np.uint8)
            total_pixels = mask_arr.size
            positive_pixels = int(mask_arr.sum())
            positive_ratio = positive_pixels / total_pixels if total_pixels > 0 else 0
        
        return {
            'height': height,
            'width': width,
            'total_pixels': total_pixels,
            'positive_pixels': positive_pixels,
            'positive_ratio': positive_ratio,
            'valid': True
        }
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return {'valid': False}


def assign_splits(case_ids: List[str], seed: int = SEED) -> Dict[str, str]:
    """Assign train/val/test splits by case_id."""
    random.seed(seed)
    shuffled = case_ids.copy()
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


def process_dataset(data_root: Path, dataset_name: str, config: Dict) -> List[Dict]:
    """Process a single dataset and return list of records."""
    image_dir = data_root / config['image_dir']
    mask_dir = data_root / config['mask_dir']
    
    if not image_dir.exists():
        print(f"Warning: {image_dir} does not exist, skipping {dataset_name}")
        return []
    
    # Find all images
    image_files = sorted(list(image_dir.glob('*.png')))
    print(f"Found {len(image_files)} images in {dataset_name}")
    
    records = []
    case_ids = []
    
    for image_path in image_files:
        # case_id = filename without extension
        case_id = image_path.stem
        mask_path = mask_dir / image_path.name
        
        if not mask_path.exists():
            print(f"Warning: mask not found for {image_path.name}")
            continue
        
        # Get image info
        info = get_image_info(image_path, mask_path)
        if not info['valid']:
            continue
        
        # Sanity check: skip all-black or all-white masks
        if info['positive_ratio'] == 0 or info['positive_ratio'] == 1:
            print(f"Warning: skipping {case_id} (positive_ratio={info['positive_ratio']:.4f})")
            continue
        
        case_ids.append(case_id)
        
        # Store relative paths
        records.append({
            'dataset': dataset_name,
            'modality': config['modality'],
            'source': config['source'],
            'case_id': case_id,
            'image_path': str(image_path.relative_to(data_root)),
            'mask_path': str(mask_path.relative_to(data_root)),
            'height': info['height'],
            'width': info['width'],
            'total_pixels': info['total_pixels'],
            'positive_pixels': info['positive_pixels'],
            'positive_ratio': info['positive_ratio'],
        })
    
    # Assign splits
    splits = assign_splits(case_ids)
    for record in records:
        record['split'] = splits[record['case_id']]
    
    return records


def compute_statistics(df: pd.DataFrame) -> Dict:
    """Compute dataset statistics."""
    stats = {}
    
    for dataset in df['dataset'].unique():
        subset = df[df['dataset'] == dataset]
        stats[dataset] = {
            'total_samples': len(subset),
            'train_samples': len(subset[subset['split'] == 'train']),
            'val_samples': len(subset[subset['split'] == 'val']),
            'test_samples': len(subset[subset['split'] == 'test']),
            'resolution': {
                'height_mean': float(subset['height'].mean()),
                'height_std': float(subset['height'].std()),
                'width_mean': float(subset['width'].mean()),
                'width_std': float(subset['width'].std()),
            },
            'mask_foreground': {
                'positive_ratio_mean': float(subset['positive_ratio'].mean()),
                'positive_ratio_std': float(subset['positive_ratio'].std()),
                'positive_ratio_min': float(subset['positive_ratio'].min()),
                'positive_ratio_max': float(subset['positive_ratio'].max()),
            }
        }
    
    # Overall statistics
    stats['overall'] = {
        'total_samples': len(df),
        'train_samples': len(df[df['split'] == 'train']),
        'val_samples': len(df[df['split'] == 'val']),
        'test_samples': len(df[df['split'] == 'test']),
    }
    
    return stats


def generate_previews(data_root: Path, df: pd.DataFrame, output_dir: Path, n_samples: int = 20):
    """Generate overlay preview images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for dataset in df['dataset'].unique():
        subset = df[df['dataset'] == dataset]
        n = min(n_samples, len(subset))
        samples = subset.sample(n=n, random_state=SEED)
        
        for idx, row in samples.iterrows():
            image_path = data_root / row['image_path']
            mask_path = data_root / row['mask_path']
            
            try:
                # Load image and mask
                image = np.array(Image.open(image_path).convert('RGB'))
                mask = np.array(Image.open(mask_path).convert('L'))
                
                # Binarize mask
                if mask.max() > 1:
                    mask = (mask > 127).astype(np.uint8) * 255
                
                # Create overlay
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                axes[0].imshow(image)
                axes[0].set_title(f'Image: {row["case_id"]}')
                axes[0].axis('off')
                
                axes[1].imshow(mask, cmap='gray')
                axes[1].set_title('Mask')
                axes[1].axis('off')
                
                # Overlay
                overlay = image.copy()
                mask_rgb = np.zeros_like(image)
                mask_rgb[:, :, 0] = mask  # Red channel
                overlay = (overlay * 0.7 + mask_rgb * 0.3).astype(np.uint8)
                
                axes[2].imshow(overlay)
                axes[2].set_title(f'Overlay (ratio={row["positive_ratio"]:.3f})')
                axes[2].axis('off')
                
                plt.tight_layout()
                save_path = output_dir / f'{dataset}_{row["case_id"]}.png'
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                print(f"Error generating preview for {row['case_id']}: {e}")
    
    print(f"Generated previews in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess CXR datasets')
    parser.add_argument('--data-root', type=str, default='.',
                        help='Root directory containing all datasets')
    args = parser.parse_args()
    
    data_root = Path(args.data_root).resolve()
    print(f"Data root: {data_root}")
    
    # Process all datasets
    all_records = []
    for dataset_name, config in DATASETS.items():
        print(f"\nProcessing {dataset_name}...")
        records = process_dataset(data_root, dataset_name, config)
        all_records.extend(records)
        print(f"  Added {len(records)} records")
    
    if not all_records:
        print("No records found!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_records)
    
    # Reorder columns
    columns = ['dataset', 'modality', 'source', 'case_id', 'image_path', 'mask_path',
               'split', 'height', 'width', 'total_pixels', 'positive_pixels', 'positive_ratio']
    df = df[columns]
    
    # Save index.csv
    index_path = data_root / 'data' / 'index.csv'
    index_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(index_path, index=False)
    print(f"\nSaved index to {index_path}")
    print(f"Total records: {len(df)}")
    
    # Print split summary
    print("\nSplit summary:")
    print(df.groupby(['dataset', 'split']).size().unstack(fill_value=0))
    
    # Compute and save statistics
    stats = compute_statistics(df)
    stats_path = data_root / 'data' / 'stats_cxr.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved statistics to {stats_path}")
    
    # Generate previews
    preview_dir = data_root / 'data' / 'preview'
    generate_previews(data_root, df, preview_dir, n_samples=20)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
