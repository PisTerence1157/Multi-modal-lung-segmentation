import os
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ChestXrayDataset(Dataset):
    """
    Unified Chest X-ray dataset supporting multiple sources.
    
    Uses index.csv with columns:
    dataset, modality, source, case_id, image_path, mask_path, split, height, width, ...
    """
    
    def __init__(self, data_root, index_path='data/index.csv', 
                 split='train', dataset=None, transform=None, image_size=(512, 512)):
        """
        Args:
            data_root: root directory for all data paths
            index_path: relative path to index.csv
            split: one of 'train', 'val', or 'test'
            dataset: filter by dataset name (e.g., 'kaggle', 'montgomery', 'shenzhen', or None for all)
            transform: albumentations transform pipeline
            image_size: target image size (H, W)
        """
        self.data_root = Path(data_root)
        self.split = split
        self.dataset_filter = dataset
        self.image_size = image_size
        
        # Load index
        index_abs = self.data_root / index_path
        self.full_index = pd.read_csv(index_abs)
        
        # Filter by split
        self.data = self.full_index[self.full_index['split'] == split].copy()
        
        # Filter by dataset if specified
        if dataset is not None:
            self.data = self.data[self.data['dataset'] == dataset].copy()
        
        self.data = self.data.reset_index(drop=True)
        
        dataset_info = f"dataset={dataset}" if dataset else "all datasets"
        print(f"Loaded {len(self.data)} samples for {split} split ({dataset_info})")
        
        # Set transforms
        if transform is None:
            self.transform = self._get_default_transform(split)
        else:
            self.transform = transform
    
    def _get_default_transform(self, split):
        """Get default augmentation pipeline."""
        if split == 'train':
            return A.Compose([
                A.Resize(*self.image_size),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                A.Normalize(mean=[0.485], std=[0.229]),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(*self.image_size),
                A.Normalize(mean=[0.485], std=[0.229]),
                ToTensorV2(),
            ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Fetch a single sample."""
        row = self.data.iloc[idx]
        
        # Build full paths using pathlib
        image_path = self.data_root / row['image_path']
        mask_path = self.data_root / row['mask_path']
        
        # Read image & mask
        image = self._load_image(image_path)
        mask = self._load_mask(mask_path)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Ensure mask has correct shape/type
        if isinstance(mask, torch.Tensor):
            mask = mask.float().unsqueeze(0)
        else:
            mask = torch.from_numpy(mask).float().unsqueeze(0)
        
        return {
            'image': image,
            'mask': mask,
            'case_id': row['case_id'],
            'dataset': row['dataset'],
            'original_size': (row['height'], row['width'])
        }
    
    def _load_image(self, image_path):
        """Load image from disk."""
        try:
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
            
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            return image
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return np.zeros(self.image_size, dtype=np.uint8)
    
    def _load_mask(self, mask_path):
        """Load segmentation mask from disk."""
        try:
            mask = Image.open(mask_path).convert('L')
            mask = np.array(mask)
            # Binarize: 0/255 -> 0/1
            mask = (mask > 127).astype(np.float32)
            return mask
            
        except Exception as e:
            print(f"Error loading mask {mask_path}: {e}")
            return np.zeros(self.image_size, dtype=np.float32)
    
    def get_positive_weight(self):
        """Compute positive class weight for imbalance handling."""
        total_positive = self.data['positive_pixels'].sum()
        total_pixels = self.data['total_pixels'].sum()
        positive_ratio = total_positive / total_pixels
        pos_weight = (1 - positive_ratio) / positive_ratio
        return pos_weight
    
    def get_sample_info(self, idx):
        """Return detailed information of a sample."""
        row = self.data.iloc[idx]
        return {
            'case_id': row['case_id'],
            'dataset': row['dataset'],
            'modality': row['modality'],
            'source': row['source'],
            'image_path': row['image_path'],
            'mask_path': row['mask_path'],
            'size': (row['height'], row['width']),
            'positive_ratio': row['positive_ratio'],
        }


def create_data_loaders(config, batch_size=None, num_workers=None):
    """Create PyTorch data loaders for train/val/test splits."""
    # Get data_root from config or environment variable
    data_root = os.environ.get('DATA_ROOT', config.get('data_root', '.'))
    
    if batch_size is None:
        batch_size = config['training']['batch_size']
    if num_workers is None:
        num_workers = config.get('num_workers', 4)
    
    data_cfg = config.get('data', {})
    dataset_cfg = config.get('dataset', {})
    
    # Get dataset filter (None = all datasets)
    dataset_filter = data_cfg.get('dataset_filter', None)
    index_path = data_cfg.get('index_path', 'data/index.csv')
    image_size = dataset_cfg.get('image_size', [512, 512])
    
    # Common dataset kwargs
    dataset_kwargs = {
        'data_root': data_root,
        'index_path': index_path,
        'dataset': dataset_filter,
        'image_size': image_size
    }
    
    # Create datasets
    train_dataset = ChestXrayDataset(**dataset_kwargs, split='train')
    val_dataset = ChestXrayDataset(**dataset_kwargs, split='val')
    test_dataset = ChestXrayDataset(**dataset_kwargs, split='test')
    
    # Data loaders
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
    
    return train_loader, val_loader, test_loader, train_dataset.get_positive_weight()


def visualize_sample(dataset, idx):
    """Visualize one sample from the dataset."""
    import matplotlib.pyplot as plt
    
    sample = dataset[idx]
    image = sample['image']
    mask = sample['mask']
    
    if isinstance(image, torch.Tensor):
        if image.shape[0] == 1:
            image = image.squeeze(0).numpy()
        else:
            image = image.permute(1, 2, 0).numpy()
    
    if isinstance(mask, torch.Tensor):
        mask = mask.squeeze().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title(f'Image: {sample["case_id"]} ({sample["dataset"]})')
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')
    
    if len(image.shape) == 2:
        overlay = np.stack([image, image, image], axis=-1)
    else:
        overlay = image.copy()
    # Normalize for display
    overlay = (overlay - overlay.min()) / (overlay.max() - overlay.min() + 1e-8)
    overlay_display = overlay.copy()
    overlay_display[mask > 0.5] = [1, 0, 0]
    
    axes[2].imshow(overlay_display)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    info = dataset.get_sample_info(idx)
    print("Sample Info:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    
    return sample
