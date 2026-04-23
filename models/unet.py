import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """双卷积块：Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> ReLU"""
    
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """下采样块：MaxPool2d -> DoubleConv"""
    
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """上采样块：ConvTranspose2d -> DoubleConv"""
    
    def __init__(self, in_channels, out_channels, bilinear=True, dropout=0.0):
        super(Up, self).__init__()
        
        # 如果使用双线性插值，减少参数数量
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, dropout)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout)
    
    def forward(self, x1, x2):
        """
        Args:
            x1: 来自上一层的特征图
            x2: 来自编码器的skip connection
        """
        x1 = self.up(x1)
        
        # 处理尺寸不匹配的情况
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # 连接特征图
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """输出卷积层"""
    
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """经典U-Net模型"""
    
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512], 
                 bilinear=True, dropout=0.0):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            features: 每层的特征数
            bilinear: 是否使用双线性插值
            dropout: dropout比例
        """
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        
        # 编码器
        self.inc = DoubleConv(in_channels, features[0], dropout)
        self.down1 = Down(features[0], features[1], dropout)
        self.down2 = Down(features[1], features[2], dropout)
        self.down3 = Down(features[2], features[3], dropout)
        
        # 底部
        factor = 2 if bilinear else 1
        self.down4 = Down(features[3], features[3] * 2 // factor, dropout)
        
        # 解码器
        self.up1 = Up(features[3] * 2, features[3] // factor, bilinear, dropout)
        self.up2 = Up(features[3], features[2] // factor, bilinear, dropout)
        self.up3 = Up(features[2], features[1] // factor, bilinear, dropout)
        self.up4 = Up(features[1], features[0], bilinear, dropout)
        
        # 输出层
        self.outc = OutConv(features[0], out_channels)
    
    def forward(self, x):
        # 编码路径
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # 解码路径
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # 输出
        logits = self.outc(x)
        return logits
    
    def get_model_summary(self, input_size=(1, 512, 512)):
        """获取模型参数总结"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # 计算模型大小（MB）
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_all_mb = (param_size + buffer_size) / 1024**2
        
        summary = {
            'model_name': 'U-Net',
            'total_params': total_params,
            'trainable_params': trainable_params,
            'non_trainable_params': total_params - trainable_params,
            'model_size_mb': size_all_mb,
            'input_size': input_size,
            'output_size': (self.out_channels, input_size[1], input_size[2])
        }
        
        return summary
    
    def print_summary(self, input_size=(1, 512, 512)):
        """打印模型摘要"""
        summary = self.get_model_summary(input_size)
        
        print("=" * 60)
        print(f"Model: {summary['model_name']}")
        print("=" * 60)
        print(f"Input size: {summary['input_size']}")
        print(f"Output size: {summary['output_size']}")
        print("-" * 60)
        print(f"Total params: {summary['total_params']:,}")
        print(f"Trainable params: {summary['trainable_params']:,}")
        print(f"Non-trainable params: {summary['non_trainable_params']:,}")
        print(f"Model size: {summary['model_size_mb']:.2f} MB")
        print("=" * 60)

def create_unet_model(config):
    """根据配置创建U-Net模型"""
    model_config = config['model']
    
    model = UNet(
        in_channels=model_config['in_channels'],
        out_channels=model_config['out_channels'],
        features=model_config['features'],
        dropout=model_config.get('dropout', 0.0)
    )
    
    return model

# 测试函数
def test_unet():
    """测试U-Net模型"""
    model = UNet(in_channels=1, out_channels=1)
    model.eval()
    
    # 创建测试输入
    x = torch.randn(2, 1, 512, 512)
    
    # 前向传播
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # 打印模型摘要
    model.print_summary()
    
    return model, output

if __name__ == "__main__":
    test_unet()

