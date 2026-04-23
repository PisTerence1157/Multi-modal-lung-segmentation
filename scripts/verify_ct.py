#!/usr/bin/env python3
"""
Verify CT data loading and 3D model.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from datasets.ct_dataset import CTDataset2D, CTDataset3D, create_ct_data_loaders
from models import UNet, UNet3D


def test_ct_2d_dataset():
    """Test 2D CT dataset."""
    print("\n" + "="*60)
    print("Testing CT 2D Dataset")
    print("="*60)
    
    try:
        dataset = CTDataset2D(
            data_root=str(project_root),
            index_path='data/index_ct.csv',
            split='train',
            image_size=(512, 512)
        )
        
        sample = dataset[0]
        print(f"Sample shape - Image: {sample['image'].shape}, Mask: {sample['mask'].shape}")
        print(f"Case: {sample['case_id']}, Slice: {sample['slice_idx']}")
        print("OK")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ct_3d_dataset():
    """Test 3D CT dataset."""
    print("\n" + "="*60)
    print("Testing CT 3D Dataset")
    print("="*60)
    
    try:
        dataset = CTDataset3D(
            data_root=str(project_root),
            index_path='data/index_ct.csv',
            split='train',
            patch_size=(64, 64, 16),  # Small for testing
            patches_per_volume=2
        )
        
        sample = dataset[0]
        print(f"Sample shape - Image: {sample['image'].shape}, Mask: {sample['mask'].shape}")
        print(f"Case: {sample['case_id']}, Origin: {sample['patch_origin']}")
        print("OK")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_unet_2d_ct():
    """Test 2D UNet on CT slice."""
    print("\n" + "="*60)
    print("Testing 2D UNet on CT")
    print("="*60)
    
    try:
        model = UNet(in_channels=1, out_channels=1)
        model.eval()
        
        x = torch.randn(2, 1, 512, 512)
        with torch.no_grad():
            y = model(x)
        
        print(f"Input: {x.shape}, Output: {y.shape}")
        print("OK")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def test_unet_3d():
    """Test 3D UNet."""
    print("\n" + "="*60)
    print("Testing 3D UNet")
    print("="*60)
    
    try:
        model = UNet3D(in_channels=1, out_channels=1, features=[16, 32, 64, 128])
        model.eval()
        
        x = torch.randn(1, 1, 64, 64, 32)
        with torch.no_grad():
            y = model(x)
        
        print(f"Input: {x.shape}, Output: {y.shape}")
        print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
        print("OK")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    results = {}
    
    results['CT 2D Dataset'] = test_ct_2d_dataset()
    results['CT 3D Dataset'] = test_ct_3d_dataset()
    results['2D UNet on CT'] = test_unet_2d_ct()
    results['3D UNet'] = test_unet_3d()
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    all_pass = True
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}")
        if not ok:
            all_pass = False
    
    return all_pass


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
