"""
CBAM-UNet: U-Net with Convolutional Block Attention Module

Adds CBAM blocks after each encoder stage for channel + spatial attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import DoubleConv, Down, Up, OutConv
from .attention_modules import CBAMBlock


class CBAMDown(nn.Module):
    """Downsampling block with CBAM attention."""
    
    def __init__(self, in_channels, out_channels, dropout=0.0, reduction=16):
        super(CBAMDown, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout)
        )
        self.cbam = CBAMBlock(out_channels, reduction)
    
    def forward(self, x):
        x = self.maxpool_conv(x)
        x = self.cbam(x)
        return x


class CBAMDoubleConv(nn.Module):
    """Double convolution with CBAM attention."""
    
    def __init__(self, in_channels, out_channels, dropout=0.0, reduction=16):
        super(CBAMDoubleConv, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels, dropout)
        self.cbam = CBAMBlock(out_channels, reduction)
    
    def forward(self, x):
        x = self.double_conv(x)
        x = self.cbam(x)
        return x


class CBAMUNet(nn.Module):
    """U-Net with CBAM (Channel + Spatial Attention) blocks."""
    
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512],
                 bilinear=True, dropout=0.0, reduction=16):
        """
        Args:
            in_channels: number of input channels
            out_channels: number of output channels
            features: list of feature dimensions for each level
            bilinear: use bilinear upsampling
            dropout: dropout rate
            reduction: CBAM reduction ratio
        """
        super(CBAMUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        
        # Encoder with CBAM blocks
        self.inc = CBAMDoubleConv(in_channels, features[0], dropout, reduction)
        self.down1 = CBAMDown(features[0], features[1], dropout, reduction)
        self.down2 = CBAMDown(features[1], features[2], dropout, reduction)
        self.down3 = CBAMDown(features[2], features[3], dropout, reduction)
        
        # Bottleneck
        factor = 2 if bilinear else 1
        self.down4 = CBAMDown(features[3], features[3] * 2 // factor, dropout, reduction)
        
        # Decoder (standard U-Net upsampling)
        self.up1 = Up(features[3] * 2, features[3] // factor, bilinear, dropout)
        self.up2 = Up(features[3], features[2] // factor, bilinear, dropout)
        self.up3 = Up(features[2], features[1] // factor, bilinear, dropout)
        self.up4 = Up(features[1], features[0], bilinear, dropout)
        
        # Output
        self.outc = OutConv(features[0], out_channels)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Collect CBAM attention maps from encoder stages
        # Each stage has both channel and spatial attention
        self.attention_weights = [
            {
                'channel': self.inc.cbam.last_channel_attention,
                'spatial': self.inc.cbam.last_spatial_attention,
            },
            {
                'channel': self.down1.cbam.last_channel_attention,
                'spatial': self.down1.cbam.last_spatial_attention,
            },
            {
                'channel': self.down2.cbam.last_channel_attention,
                'spatial': self.down2.cbam.last_spatial_attention,
            },
            {
                'channel': self.down3.cbam.last_channel_attention,
                'spatial': self.down3.cbam.last_spatial_attention,
            },
            {
                'channel': self.down4.cbam.last_channel_attention,
                'spatial': self.down4.cbam.last_spatial_attention,
            },
        ]
        
        # Output
        logits = self.outc(x)
        return logits
    
    def get_model_summary(self, input_size=(1, 512, 512)):
        """Get model summary."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        param_size = sum(p.nelement() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.buffers())
        size_mb = (param_size + buffer_size) / 1024**2
        
        return {
            'model_name': 'CBAM-UNet',
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': size_mb,
        }


def create_cbam_unet_model(config):
    """Create CBAM-UNet from config."""
    model_config = config['model']
    return CBAMUNet(
        in_channels=model_config.get('in_channels', 1),
        out_channels=model_config.get('out_channels', 1),
        features=model_config.get('features', [64, 128, 256, 512]),
        dropout=model_config.get('dropout', 0.0),
        bilinear=model_config.get('bilinear', True),
        reduction=model_config.get('cbam_reduction', 16)
    )


def test_cbam_unet():
    """Test CBAM-UNet."""
    model = CBAMUNet(in_channels=1, out_channels=1)
    model.eval()
    
    x = torch.randn(2, 1, 512, 512)
    with torch.no_grad():
        y = model(x)
    
    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    return model


if __name__ == "__main__":
    test_cbam_unet()
