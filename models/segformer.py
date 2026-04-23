"""
SegFormer for Medical Image Segmentation

Wrapper around HuggingFace SegFormer for single-channel medical images.

Reference: Xie et al., "SegFormer: Simple and Efficient Design for Semantic 
Segmentation with Transformers", NeurIPS 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import SegformerForSemanticSegmentation, SegformerConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers library not found. Install with: pip install transformers")


class SegFormerWrapper(nn.Module):
    """
    SegFormer wrapper for medical image segmentation.
    
    Handles:
    - 1-channel grayscale input -> 3-channel (repeat)
    - ImageNet normalization (for pretrained weights)
    - Output upsampling to match input resolution
    """
    
    # ImageNet normalization (used by pretrained SegFormer)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    def __init__(self, 
                 in_channels=1, 
                 out_channels=1,
                 pretrained_model="nvidia/segformer-b0-finetuned-ade-512-512",
                 use_imagenet_norm=True):
        """
        Args:
            in_channels: number of input channels (1 for grayscale)
            out_channels: number of output classes (1 for binary segmentation)
            pretrained_model: HuggingFace model name or path
            use_imagenet_norm: whether to apply ImageNet normalization
        """
        super(SegFormerWrapper, self).__init__()
        
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers library required for SegFormer")
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_imagenet_norm = use_imagenet_norm
        
        # Load pretrained SegFormer and modify for our task
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_model,
            num_labels=out_channels,
            ignore_mismatched_sizes=True
        )
        
        # Register normalization buffers
        self.register_buffer('mean', torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1))
    
    def _preprocess(self, x):
        """
        Preprocess input:
        1. Repeat 1-channel to 3-channel
        2. Apply ImageNet normalization if enabled
        """
        # Handle single channel input
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # Denormalize from dataset norm (assumed 0.485, 0.229 for grayscale)
        # Then apply ImageNet normalization
        if self.use_imagenet_norm:
            # Assuming input is already normalized with mean=0.485, std=0.229
            # First denormalize
            x = x * 0.229 + 0.485
            # Then apply ImageNet normalization per channel
            x = (x - self.mean) / self.std
        
        return x
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: input tensor [B, C, H, W]
            
        Returns:
            logits: output tensor [B, out_channels, H, W]
        """
        input_size = x.shape[2:]
        
        # Preprocess
        x = self._preprocess(x)
        
        # Forward through SegFormer
        outputs = self.segformer(pixel_values=x)
        logits = outputs.logits
        
        # Upsample to input resolution
        logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
        
        return logits
    
    def get_model_summary(self, input_size=(1, 512, 512)):
        """Get model summary."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        param_size = sum(p.nelement() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.buffers())
        size_mb = (param_size + buffer_size) / 1024**2
        
        return {
            'model_name': 'SegFormer',
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': size_mb,
        }


class SegFormerFromScratch(nn.Module):
    """
    SegFormer built from scratch (no pretrained weights).
    Useful when HuggingFace model download is not available.
    """
    
    def __init__(self, in_channels=1, out_channels=1, embed_dims=[32, 64, 160, 256],
                 num_heads=[1, 2, 5, 8], depths=[2, 2, 2, 2]):
        super(SegFormerFromScratch, self).__init__()
        
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers library required for SegFormer")
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Create config for SegFormer-B0 equivalent
        config = SegformerConfig(
            num_channels=3,  # Will repeat grayscale to 3 channels
            num_labels=out_channels,
            hidden_sizes=embed_dims,
            num_attention_heads=num_heads,
            depths=depths,
        )
        
        self.segformer = SegformerForSemanticSegmentation(config)
        
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, x):
        input_size = x.shape[2:]
        
        # Repeat to 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # Normalize
        x = x * 0.229 + 0.485  # denormalize from dataset
        x = (x - self.mean) / self.std  # ImageNet normalize
        
        # Forward
        outputs = self.segformer(pixel_values=x)
        logits = outputs.logits
        
        # Upsample
        logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
        
        return logits


def create_segformer_model(config):
    """Create SegFormer from config."""
    model_config = config.get('model', {})
    
    pretrained = model_config.get('pretrained', True)
    pretrained_model = model_config.get(
        'pretrained_model', 
        'nvidia/segformer-b0-finetuned-ade-512-512'
    )
    
    if pretrained:
        return SegFormerWrapper(
            in_channels=model_config.get('in_channels', 1),
            out_channels=model_config.get('out_channels', 1),
            pretrained_model=pretrained_model,
            use_imagenet_norm=model_config.get('use_imagenet_norm', True)
        )
    else:
        return SegFormerFromScratch(
            in_channels=model_config.get('in_channels', 1),
            out_channels=model_config.get('out_channels', 1),
        )


def test_segformer():
    """Test SegFormer."""
    if not HAS_TRANSFORMERS:
        print("Skipping SegFormer test: transformers not installed")
        return None
    
    print("Testing SegFormer...")
    model = SegFormerWrapper(in_channels=1, out_channels=1)
    model.eval()
    
    x = torch.randn(2, 1, 512, 512)
    with torch.no_grad():
        y = model(x)
    
    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


if __name__ == "__main__":
    test_segformer()
