from .unet import UNet, create_unet_model
from .attention_unet import AttentionUNet, create_attention_unet_model
from .se_unet import SEUNet, create_se_unet_model
from .cbam_unet import CBAMUNet, create_cbam_unet_model
from .attention_modules import SEBlock, CBAMBlock
from .unet3d import UNet3D, create_unet3d_model

# SegFormer requires transformers library
try:
    from .segformer import SegFormerWrapper, create_segformer_model
    HAS_SEGFORMER = True
except ImportError:
    HAS_SEGFORMER = False

__all__ = [
    'UNet', 'create_unet_model',
    'AttentionUNet', 'create_attention_unet_model',
    'SEUNet', 'create_se_unet_model',
    'CBAMUNet', 'create_cbam_unet_model',
    'SEBlock', 'CBAMBlock',
    'UNet3D', 'create_unet3d_model',
]

if HAS_SEGFORMER:
    __all__.extend(['SegFormerWrapper', 'create_segformer_model'])


def get_model(config):
    """Get model by name from config."""
    model_name = config['model']['name'].lower()
    
    if model_name == 'unet':
        return create_unet_model(config)
    elif model_name == 'attention_unet':
        return create_attention_unet_model(config)
    elif model_name == 'se_unet':
        return create_se_unet_model(config)
    elif model_name == 'cbam_unet':
        return create_cbam_unet_model(config)
    elif model_name == 'segformer':
        if not HAS_SEGFORMER:
            raise ImportError("SegFormer requires transformers library. Install with: pip install transformers")
        return create_segformer_model(config)
    elif model_name == 'unet3d':
        return create_unet3d_model(config)
    else:
        raise ValueError(f"Unknown model: {model_name}")
