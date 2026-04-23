#!/usr/bin/env python3
"""
Verify data loading for all CXR datasets.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datasets.chest_xray_dataset import ChestXrayDataset, create_data_loaders
from engine.trainer import load_config


def verify_dataset(data_root, dataset_name=None):
    """Verify a dataset can be loaded."""
    print(f"\n{'='*60}")
    print(f"Testing dataset: {dataset_name or 'all'}")
    print('='*60)
    
    try:
        # Create dataset
        train_ds = ChestXrayDataset(
            data_root=data_root,
            index_path='data/index.csv',
            split='train',
            dataset=dataset_name,
            image_size=(512, 512)
        )
        
        val_ds = ChestXrayDataset(
            data_root=data_root,
            index_path='data/index.csv',
            split='val',
            dataset=dataset_name,
            image_size=(512, 512)
        )
        
        test_ds = ChestXrayDataset(
            data_root=data_root,
            index_path='data/index.csv',
            split='test',
            dataset=dataset_name,
            image_size=(512, 512)
        )
        
        print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
        
        # Load a sample
        sample = train_ds[0]
        print(f"Sample shape - Image: {sample['image'].shape}, Mask: {sample['mask'].shape}")
        print(f"Sample info: case_id={sample['case_id']}, dataset={sample['dataset']}")
        
        # Check positive weight
        pos_weight = train_ds.get_positive_weight()
        print(f"Positive weight: {pos_weight:.4f}")
        
        print("OK")
        return True
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    data_root = Path(__file__).parent.parent
    
    # Test each dataset
    datasets = [None, 'kaggle', 'montgomery', 'shenzhen']
    results = {}
    
    for ds_name in datasets:
        results[ds_name or 'all'] = verify_dataset(data_root, ds_name)
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print('='*60)
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}")
    
    # Test config loading
    print(f"\n{'='*60}")
    print("Testing config loading")
    print('='*60)
    
    try:
        config = load_config(
            config_path=data_root / 'configs/unet.yaml',
            dataset_config=data_root / 'configs/datasets/kaggle.yaml'
        )
        print(f"Model: {config['model']['name']}")
        print(f"Dataset filter: {config['data'].get('dataset_filter')}")
        print(f"Dataset name: {config['data'].get('dataset_name')}")
        print("Config loading: OK")
    except Exception as e:
        print(f"Config loading: FAILED - {e}")


if __name__ == '__main__':
    main()
