"""
3D U-Net for volumetric medical image segmentation.

Reference: Cicek et al., "3D U-Net: Learning Dense Volumetric Segmentation 
from Sparse Annotation", MICCAI 2016
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3D(nn.Module):
    """Double 3D convolution block."""
    
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(DoubleConv3D, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down3D(nn.Module):
    """Downsampling block for 3D U-Net."""
    
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(Down3D, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels, dropout)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    """Upsampling block for 3D U-Net."""
    
    def __init__(self, in_channels, out_channels, trilinear=True, dropout=0.0):
        super(Up3D, self).__init__()
        
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels, dropout)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels, dropout)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle size mismatch
        diffD = x2.size()[2] - x1.size()[2]
        diffH = x2.size()[3] - x1.size()[3]
        diffW = x2.size()[4] - x1.size()[4]
        
        x1 = F.pad(x1, [diffW // 2, diffW - diffW // 2,
                        diffH // 2, diffH - diffH // 2,
                        diffD // 2, diffD - diffD // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """3D U-Net for volumetric segmentation."""
    
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256],
                 trilinear=True, dropout=0.0):
        """
        Args:
            in_channels: number of input channels
            out_channels: number of output channels
            features: feature dimensions at each level (smaller than 2D due to memory)
            trilinear: use trilinear upsampling
            dropout: dropout rate
        """
        super(UNet3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.trilinear = trilinear
        
        # Encoder
        self.inc = DoubleConv3D(in_channels, features[0], dropout)
        self.down1 = Down3D(features[0], features[1], dropout)
        self.down2 = Down3D(features[1], features[2], dropout)
        self.down3 = Down3D(features[2], features[3], dropout)
        
        # Bottleneck
        factor = 2 if trilinear else 1
        self.down4 = Down3D(features[3], features[3] * 2 // factor, dropout)
        
        # Decoder
        self.up1 = Up3D(features[3] * 2, features[3] // factor, trilinear, dropout)
        self.up2 = Up3D(features[3], features[2] // factor, trilinear, dropout)
        self.up3 = Up3D(features[2], features[1] // factor, trilinear, dropout)
        self.up4 = Up3D(features[1], features[0], trilinear, dropout)
        
        # Output
        self.outc = nn.Conv3d(features[0], out_channels, kernel_size=1)
    
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
        
        # Output
        logits = self.outc(x)
        return logits
    
    def get_model_summary(self, input_size=(1, 128, 128, 32)):
        """Get model summary."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        param_size = sum(p.nelement() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.buffers())
        size_mb = (param_size + buffer_size) / 1024**2
        
        return {
            'model_name': '3D U-Net',
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': size_mb,
        }


def create_unet3d_model(config):
    """Create 3D U-Net from config."""
    model_config = config.get('model', {})
    return UNet3D(
        in_channels=model_config.get('in_channels', 1),
        out_channels=model_config.get('out_channels', 1),
        features=model_config.get('features', [32, 64, 128, 256]),
        dropout=model_config.get('dropout', 0.0),
        trilinear=model_config.get('trilinear', True)
    )


def test_unet3d():
    """Test 3D U-Net."""
    model = UNet3D(in_channels=1, out_channels=1)
    model.eval()
    
    # Test with small input (memory efficient)
    x = torch.randn(1, 1, 64, 64, 32)
    
    with torch.no_grad():
        y = model(x)
    
    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


if __name__ == "__main__":
    test_unet3d()
