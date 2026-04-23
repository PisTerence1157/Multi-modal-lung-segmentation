#!/usr/bin/env python3
"""
Verify all models can be instantiated and run forward pass.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from models import UNet, AttentionUNet, SEUNet, CBAMUNet, get_model
from models import HAS_SEGFORMER
if HAS_SEGFORMER:
    from models import SegFormerWrapper
from engine.trainer import load_config


def test_model(model_class, name):
    """Test a single model."""
    print(f"\nTesting {name}...")
    try:
        model = model_class(in_channels=1, out_channels=1)
        model.eval()
        
        x = torch.randn(2, 1, 512, 512)
        with torch.no_grad():
            y = model(x)
        
        params = sum(p.numel() for p in model.parameters())
        print(f"  Input: {x.shape}")
        print(f"  Output: {y.shape}")
        print(f"  Params: {params:,}")
        
        assert y.shape == (2, 1, 512, 512), f"Output shape mismatch: {y.shape}"
        print(f"  Status: OK")
        return True
    except Exception as e:
        print(f"  Status: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_loading():
    """Test model creation from config."""
    print("\n" + "="*60)
    print("Testing config-based model creation")
    print("="*60)
    
    configs = [
        ('unet', 'configs/unet.yaml'),
        ('attention_unet', 'configs/attention_unet.yaml'),
        ('se_unet', 'configs/se_unet.yaml'),
        ('cbam_unet', 'configs/cbam_unet.yaml'),
    ]
    
    if HAS_SEGFORMER:
        configs.append(('segformer', 'configs/segformer.yaml'))
    
    results = {}
    for name, config_path in configs:
        try:
            config = load_config(project_root / config_path)
            model = get_model(config)
            
            x = torch.randn(1, 1, 512, 512)
            with torch.no_grad():
                y = model(x)
            
            print(f"  {name}: OK (params={sum(p.numel() for p in model.parameters()):,})")
            results[name] = True
        except Exception as e:
            print(f"  {name}: FAILED - {e}")
            results[name] = False
    
    return results


def main():
    print("="*60)
    print("Model Verification")
    print("="*60)
    
    # Test each model class
    models = [
        (UNet, 'UNet'),
        (AttentionUNet, 'Attention-UNet'),
        (SEUNet, 'SE-UNet'),
        (CBAMUNet, 'CBAM-UNet'),
    ]
    
    if HAS_SEGFORMER:
        models.append((SegFormerWrapper, 'SegFormer'))
    
    results = {}
    for model_class, name in models:
        results[name] = test_model(model_class, name)
    
    # Test config loading
    config_results = test_config_loading()
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    all_pass = True
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}")
        if not ok:
            all_pass = False
    
    print("\nConfig-based loading:")
    for name, ok in config_results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}")
        if not ok:
            all_pass = False
    
    return all_pass


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
