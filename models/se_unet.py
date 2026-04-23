"""
SE-UNet: U-Net with Squeeze-and-Excitation Blocks

Adds SE blocks after each encoder stage to enhance channel attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import DoubleConv, Down, Up, OutConv
from .attention_modules import SEBlock


class SEDown(nn.Module):
    """Downsampling block with SE attention."""
    
    def __init__(self, in_channels, out_channels, dropout=0.0, reduction=16):
        super(SEDown, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout)
        )
        self.se = SEBlock(out_channels, reduction)
    
    def forward(self, x):
        x = self.maxpool_conv(x)
        x = self.se(x)
        return x


class SEDoubleConv(nn.Module):
    """Double convolution with SE attention."""
    
    def __init__(self, in_channels, out_channels, dropout=0.0, reduction=16):
        super(SEDoubleConv, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels, dropout)
        self.se = SEBlock(out_channels, reduction)
    
    def forward(self, x):
        x = self.double_conv(x)
        x = self.se(x)
        return x


class SEUNet(nn.Module):
    """U-Net with Squeeze-and-Excitation blocks."""
    
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512],
                 bilinear=True, dropout=0.0, reduction=16):
        """
        Args:
            in_channels: number of input channels
            out_channels: number of output channels
            features: list of feature dimensions for each level
            bilinear: use bilinear upsampling
            dropout: dropout rate
            reduction: SE reduction ratio
        """
        super(SEUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        
        # Encoder with SE blocks
        self.inc = SEDoubleConv(in_channels, features[0], dropout, reduction)
        self.down1 = SEDown(features[0], features[1], dropout, reduction)
        self.down2 = SEDown(features[1], features[2], dropout, reduction)
        self.down3 = SEDown(features[2], features[3], dropout, reduction)
        
        # Bottleneck
        factor = 2 if bilinear else 1
        self.down4 = SEDown(features[3], features[3] * 2 // factor, dropout, reduction)
        
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
        
        # Collect SE attention maps from encoder stages
        self.attention_weights = [
            self.inc.se.last_attention_map,
            self.down1.se.last_attention_map,
            self.down2.se.last_attention_map,
            self.down3.se.last_attention_map,
            self.down4.se.last_attention_map,
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
            'model_name': 'SE-UNet',
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': size_mb,
        }


def create_se_unet_model(config):
    """Create SE-UNet from config."""
    model_config = config['model']
    return SEUNet(
        in_channels=model_config.get('in_channels', 1),
        out_channels=model_config.get('out_channels', 1),
        features=model_config.get('features', [64, 128, 256, 512]),
        dropout=model_config.get('dropout', 0.0),
        bilinear=model_config.get('bilinear', True),
        reduction=model_config.get('se_reduction', 16)
    )


def test_se_unet():
    """Test SE-UNet."""
    model = SEUNet(in_channels=1, out_channels=1)
    model.eval()
    
    x = torch.randn(2, 1, 512, 512)
    with torch.no_grad():
        y = model(x)
    
    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    return model


if __name__ == "__main__":
    test_se_unet()
