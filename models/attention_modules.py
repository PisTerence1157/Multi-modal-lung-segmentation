"""
Attention Modules for Medical Image Segmentation

Includes:
- SE (Squeeze-and-Excitation) Block
- CBAM (Convolutional Block Attention Module)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    
    Reference: Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018
    
    Applies channel-wise attention by:
    1. Global average pooling (squeeze)
    2. FC -> ReLU -> FC -> Sigmoid (excitation)
    3. Channel-wise multiplication
    """
    
    def __init__(self, channels, reduction=16):
        """
        Args:
            channels: number of input channels
            reduction: reduction ratio for the bottleneck
        """
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Store attention weights for visualization
        self.last_attention_map = y.detach()
        # Scale
        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (part of CBAM)
    
    Uses both average-pooling and max-pooling for richer features.
    """
    
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        attn = self.sigmoid(avg_out + max_out)
        self.last_attention_map = attn.detach()
        return attn


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (part of CBAM)
    
    Uses channel-wise pooling to generate spatial attention map.
    """
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Channel-wise pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # Concatenate and convolve
        concat = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(concat))
        self.last_attention_map = attn.detach()
        return attn


class CBAMBlock(nn.Module):
    """
    Convolutional Block Attention Module
    
    Reference: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018
    
    Sequential application of Channel and Spatial attention.
    """
    
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        """
        Args:
            channels: number of input channels
            reduction: reduction ratio for channel attention
            spatial_kernel: kernel size for spatial attention
        """
        super(CBAMBlock, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel)
    
    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca
        # Spatial attention
        sa = self.spatial_attention(x)
        x = x * sa
        # Store both attention maps for visualization
        self.last_channel_attention = ca.detach()
        self.last_spatial_attention = sa.detach()
        return x


# Test functions
def test_se_block():
    """Test SE Block."""
    block = SEBlock(64)
    x = torch.randn(2, 64, 32, 32)
    y = block(x)
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"
    print(f"SE Block: input {x.shape} -> output {y.shape}")
    return True


def test_cbam_block():
    """Test CBAM Block."""
    block = CBAMBlock(64)
    x = torch.randn(2, 64, 32, 32)
    y = block(x)
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"
    print(f"CBAM Block: input {x.shape} -> output {y.shape}")
    return True


if __name__ == "__main__":
    test_se_block()
    test_cbam_block()
    print("All tests passed!")
