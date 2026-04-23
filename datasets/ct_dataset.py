"""
CT Dataset for 2D and 3D segmentation.

Supports:
- 2D mode: slice-by-slice training/inference
- 3D mode: patch-based training with sliding window inference
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset


def load_analyze_volume(hdr_path: Path) -> np.ndarray:
    """Load Analyze format volume."""
    img = nib.load(str(hdr_path))
    return img.get_fdata().astype(np.float32)


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert mask to binary (0/1). Handles -1024/-1023 encoding."""
    unique_vals = np.unique(mask)
    if len(unique_vals) == 2 and unique_vals[0] < 0:
        return (mask > mask.min()).astype(np.float32)
    return (mask > 0).astype(np.float32)


def normalize_ct(volume: np.ndarray, window_center: float = -600, 
                 window_width: float = 1500) -> np.ndarray:
    """
    Normalize CT volume using windowing (common for lung CT).
    
    Default window: center=-600, width=1500 (lung window)
    """
    min_val = window_center - window_width / 2
    max_val = window_center + window_width / 2
    
    volume = np.clip(volume, min_val, max_val)
    volume = (volume - min_val) / (max_val - min_val)
    
    return volume


class CTDataset2D(Dataset):
    """
    CT Dataset for 2D slice-by-slice training.
    
    Each sample is a single 2D slice from a CT volume.
    Volumes are cached in memory for fast access.
    """
    
    def __init__(self, data_root: str, index_path: str = 'data/index_ct.csv',
                 split: str = 'train', image_size: Tuple[int, int] = (512, 512),
                 window_center: float = -600, window_width: float = 1500,
                 skip_empty: bool = True, min_foreground_ratio: float = 0.001,
                 cache_volumes: bool = True):
        """
        Args:
            data_root: root data directory
            index_path: path to index_ct.csv
            split: train/val/test
            image_size: target size (H, W)
            window_center: CT window center (HU)
            window_width: CT window width (HU)
            skip_empty: skip slices with no foreground
            min_foreground_ratio: minimum foreground ratio to include slice
            cache_volumes: whether to cache volumes in memory (much faster)
        """
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.window_center = window_center
        self.window_width = window_width
        self.cache_volumes = cache_volumes
        
        # Load index
        index_abs = self.data_root / index_path
        full_index = pd.read_csv(index_abs)
        self.case_index = full_index[full_index['split'] == split].reset_index(drop=True)
        
        # Volume cache: {case_idx: (image_normalized, mask_binary)}
        self.volume_cache = {}
        
        # Build slice index (and optionally cache volumes)
        self.slice_index = self._build_slice_index(skip_empty, min_foreground_ratio)
        
        print(f"CT 2D Dataset: {len(self.case_index)} cases, {len(self.slice_index)} slices ({split})")
        if self.cache_volumes:
            print(f"  Cached {len(self.volume_cache)} volumes in memory")
    
    def _load_and_cache_volume(self, case_idx: int, row) -> Tuple[np.ndarray, np.ndarray]:
        """Load volume and cache it, or return from cache."""
        if case_idx in self.volume_cache:
            return self.volume_cache[case_idx]
        
        # Load volumes
        image_path = self.data_root / row['image_path']
        mask_path = self.data_root / row['mask_path']
        
        image = load_analyze_volume(image_path)
        mask = load_analyze_volume(mask_path)
        mask = binarize_mask(mask)
        
        # Normalize CT (do it once, not per-slice)
        image = normalize_ct(image, self.window_center, self.window_width)
        
        if self.cache_volumes:
            self.volume_cache[case_idx] = (image, mask)
        
        return image, mask
    
    def _build_slice_index(self, skip_empty: bool, min_ratio: float) -> List[Dict]:
        """Build index of (case_id, slice_idx) pairs."""
        slice_index = []
        
        for case_idx, row in self.case_index.iterrows():
            case_id = row['case_id']
            depth = int(row['depth'])
            
            if skip_empty or self.cache_volumes:
                # Load and cache volume
                image, mask = self._load_and_cache_volume(case_idx, row)
                
                if skip_empty:
                    for z in range(depth):
                        slice_mask = mask[:, :, z]
                        ratio = slice_mask.sum() / slice_mask.size
                        if ratio >= min_ratio:
                            slice_index.append({
                                'case_idx': case_idx,
                                'case_id': case_id,
                                'slice_idx': z,
                                'foreground_ratio': ratio
                            })
                else:
                    for z in range(depth):
                        slice_index.append({
                            'case_idx': case_idx,
                            'case_id': case_id,
                            'slice_idx': z,
                            'foreground_ratio': None
                        })
            else:
                for z in range(depth):
                    slice_index.append({
                        'case_idx': case_idx,
                        'case_id': case_id,
                        'slice_idx': z,
                        'foreground_ratio': None
                    })
        
        return slice_index
    
    def __len__(self):
        return len(self.slice_index)
    
    def __getitem__(self, idx):
        info = self.slice_index[idx]
        case_idx = info['case_idx']
        row = self.case_index.iloc[case_idx]
        
        # Get volume from cache or load
        image, mask = self._load_and_cache_volume(case_idx, row)
        
        # Get slice
        z = info['slice_idx']
        image_slice = image[:, :, z].copy()
        mask_slice = mask[:, :, z].copy()
        
        # Resize if needed
        if self.image_size != (image_slice.shape[0], image_slice.shape[1]):
            import cv2
            image_slice = cv2.resize(image_slice, self.image_size[::-1])
            mask_slice = cv2.resize(mask_slice, self.image_size[::-1], interpolation=cv2.INTER_NEAREST)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_slice).float().unsqueeze(0)  # [1, H, W]
        mask_tensor = torch.from_numpy(mask_slice).float().unsqueeze(0)  # [1, H, W]
        
        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'case_id': info['case_id'],
            'slice_idx': z,
        }


class CTDataset3D(Dataset):
    """
    CT Dataset for 3D patch-based training.
    
    Each sample is a 3D patch from a CT volume.
    """
    
    def __init__(self, data_root: str, index_path: str = 'data/index_ct.csv',
                 split: str = 'train', patch_size: Tuple[int, int, int] = (128, 128, 32),
                 window_center: float = -600, window_width: float = 1500,
                 patches_per_volume: int = 4, foreground_ratio: float = 0.5):
        """
        Args:
            data_root: root data directory
            index_path: path to index_ct.csv
            split: train/val/test
            patch_size: (D, H, W) patch size
            window_center: CT window center
            window_width: CT window width
            patches_per_volume: number of patches to sample per volume per epoch
            foreground_ratio: ratio of patches that should contain foreground
        """
        self.data_root = Path(data_root)
        self.split = split
        self.patch_size = patch_size
        self.window_center = window_center
        self.window_width = window_width
        self.patches_per_volume = patches_per_volume
        self.foreground_ratio = foreground_ratio
        
        # Load index
        index_abs = self.data_root / index_path
        full_index = pd.read_csv(index_abs)
        self.case_index = full_index[full_index['split'] == split].reset_index(drop=True)
        
        # Preload foreground locations for efficient sampling
        self._preload_foreground_info()
        
        print(f"CT 3D Dataset: {len(self.case_index)} cases, patch_size={patch_size} ({split})")
    
    def _preload_foreground_info(self):
        """Preload foreground voxel locations for foreground-biased sampling."""
        self.foreground_coords = {}
        
        for idx, row in self.case_index.iterrows():
            mask_path = self.data_root / row['mask_path']
            mask = load_analyze_volume(mask_path)
            mask = binarize_mask(mask)
            
            # Get foreground coordinates
            coords = np.argwhere(mask > 0)
            self.foreground_coords[idx] = coords if len(coords) > 0 else None
    
    def __len__(self):
        return len(self.case_index) * self.patches_per_volume
    
    def __getitem__(self, idx):
        # Determine which volume and which patch
        vol_idx = idx // self.patches_per_volume
        patch_idx = idx % self.patches_per_volume
        
        row = self.case_index.iloc[vol_idx]
        
        # Load volume
        image_path = self.data_root / row['image_path']
        mask_path = self.data_root / row['mask_path']
        
        image = load_analyze_volume(image_path)
        mask = load_analyze_volume(mask_path)
        mask = binarize_mask(mask)
        
        # Normalize
        image = normalize_ct(image, self.window_center, self.window_width)
        
        # Sample patch location
        patch = self._sample_patch(image, mask, vol_idx)
        
        # Convert to tensor [C, D, H, W]
        image_tensor = torch.from_numpy(patch['image']).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(patch['mask']).float().unsqueeze(0)
        
        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'case_id': row['case_id'],
            'patch_origin': patch['origin'],
        }
    
    def _sample_patch(self, image: np.ndarray, mask: np.ndarray, vol_idx: int) -> Dict:
        """Sample a patch from the volume."""
        h, w, d = image.shape
        ph, pw, pd = self.patch_size
        
        # Decide whether to sample foreground-centered patch
        use_foreground = (np.random.random() < self.foreground_ratio and 
                          self.foreground_coords[vol_idx] is not None)
        
        if use_foreground and len(self.foreground_coords[vol_idx]) > 0:
            # Sample a foreground voxel as center
            coords = self.foreground_coords[vol_idx]
            center = coords[np.random.randint(len(coords))]
            
            # Calculate patch origin (centered on foreground voxel)
            origin = [
                max(0, min(center[0] - ph // 2, h - ph)),
                max(0, min(center[1] - pw // 2, w - pw)),
                max(0, min(center[2] - pd // 2, d - pd)),
            ]
        else:
            # Random patch
            origin = [
                np.random.randint(0, max(1, h - ph + 1)),
                np.random.randint(0, max(1, w - pw + 1)),
                np.random.randint(0, max(1, d - pd + 1)),
            ]
        
        # Handle case where volume is smaller than patch
        origin = [max(0, min(o, s - p)) for o, s, p in zip(origin, [h, w, d], [ph, pw, pd])]
        
        # Extract patch
        oh, ow, od = origin
        image_patch = image[oh:oh+ph, ow:ow+pw, od:od+pd]
        mask_patch = mask[oh:oh+ph, ow:ow+pw, od:od+pd]
        
        # Pad if needed
        if image_patch.shape != (ph, pw, pd):
            pad_image = np.zeros((ph, pw, pd), dtype=np.float32)
            pad_mask = np.zeros((ph, pw, pd), dtype=np.float32)
            sh = image_patch.shape
            pad_image[:sh[0], :sh[1], :sh[2]] = image_patch
            pad_mask[:sh[0], :sh[1], :sh[2]] = mask_patch
            image_patch = pad_image
            mask_patch = pad_mask
        
        return {
            'image': image_patch,
            'mask': mask_patch,
            'origin': origin
        }


def create_ct_data_loaders(config, mode='2d', batch_size=None, num_workers=None):
    """
    Create CT data loaders.
    
    Args:
        config: configuration dict
        mode: '2d' or '3d'
        batch_size: override batch size
        num_workers: override num workers
    """
    data_root = os.environ.get('DATA_ROOT', config.get('data_root', '.'))
    
    if batch_size is None:
        batch_size = config['training']['batch_size']
    if num_workers is None:
        num_workers = config.get('num_workers', 4)
    
    ct_config = config.get('ct', {})
    
    if mode == '2d':
        DatasetClass = CTDataset2D
        dataset_kwargs = {
            'data_root': data_root,
            'index_path': 'data/index_ct.csv',
            'image_size': config.get('dataset', {}).get('image_size', [512, 512]),
            'window_center': ct_config.get('window_center', -600),
            'window_width': ct_config.get('window_width', 1500),
        }
    else:  # 3d
        DatasetClass = CTDataset3D
        dataset_kwargs = {
            'data_root': data_root,
            'index_path': 'data/index_ct.csv',
            'patch_size': ct_config.get('patch_size', [128, 128, 32]),
            'window_center': ct_config.get('window_center', -600),
            'window_width': ct_config.get('window_width', 1500),
            'patches_per_volume': ct_config.get('patches_per_volume', 4),
        }
    
    train_dataset = DatasetClass(**dataset_kwargs, split='train')
    val_dataset = DatasetClass(**dataset_kwargs, split='val')
    test_dataset = DatasetClass(**dataset_kwargs, split='test')
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader
