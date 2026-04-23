import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet import DoubleConv, Down, OutConv

class AttentionGate(nn.Module):
    """注意力门机制"""
    
    def __init__(self, F_g, F_l, F_int):
        """
        Args:
            F_g: 门控信号的通道数 (来自更深层)
            F_l: 局部特征的通道数 (来自skip connection)
            F_int: 中间特征的通道数
        """
        super(AttentionGate, self).__init__()
        
        # 门控信号的卷积
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # 局部特征的卷积
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # 输出层
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        # 激活函数
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        """
        Args:
            g: 门控信号 (来自更深层)
            x: 局部特征 (来自skip connection)
        """
        # 获取输入尺寸
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # 如果尺寸不匹配，对g进行上采样
        if g1.size()[2:] != x1.size()[2:]:
            g1 = F.interpolate(g1, size=x1.size()[2:], mode='bilinear', align_corners=False)
        
        # 相加并激活
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        # Store attention map for visualization
        self.last_attention_map = psi.detach()
        
        # 应用注意力
        return x * psi

class AttentionUp(nn.Module):
    """带注意力机制的上采样块"""

    def __init__(self, in_channels, skip_channels, out_channels, bilinear=True, dropout=0.0):
        super().__init__()
        self.bilinear = bilinear
        self.skip_channels = skip_channels

        # 上采样后门控特征的通道数（= cat 前的解码分支通道）
        g_channels = in_channels - skip_channels  # 兼容 bilinear / transposed 两种分支

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            # 直接把上采样输出设为 g_channels，避免写死 in_channels // 2 带来的不一致
            self.up = nn.ConvTranspose2d(g_channels, g_channels, kernel_size=2, stride=2)

        # 注意力门：F_g 用上采样后的门控通道，F_l 用 skip 通道
        self.attention = AttentionGate(
            F_g=g_channels,
            F_l=skip_channels,
            F_int=g_channels // 2
        )

        # 上采样后与 skip 连接，conv 的输入通道 = in_channels（= g_channels + skip_channels）
        self.conv = DoubleConv(in_channels, out_channels, dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)               # [B, g_channels, H*, W*]
        x2_att = self.attention(g=x1, x=x2)

        # 尺寸对齐
        diffY = x2_att.size(2) - x1.size(2)
        diffX = x2_att.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # 拼接后卷积
        x = torch.cat([x2_att, x1], dim=1)   # 通道 = in_channels
        return self.conv(x)


class AttentionUNet(nn.Module):
    """带注意力机制的U-Net模型"""
    
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
        super(AttentionUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        
        # 编码器 (与原始U-Net相同)
        self.inc = DoubleConv(in_channels, features[0], dropout)
        self.down1 = Down(features[0], features[1], dropout)
        self.down2 = Down(features[1], features[2], dropout)
        self.down3 = Down(features[2], features[3], dropout)
        
        # 底部
        factor = 2 if bilinear else 1
        self.down4 = Down(features[3], features[3] * 2 // factor, dropout)
        
        # 解码器 (使用注意力机制)
        factor = 2 if bilinear else 1

        self.up1 = AttentionUp(features[3] * 2, features[3], features[3] // factor, bilinear, dropout)
        self.up2 = AttentionUp(features[3],      features[2], features[2] // factor, bilinear, dropout)
        self.up3 = AttentionUp(features[2],      features[1], features[1] // factor, bilinear, dropout)
        self.up4 = AttentionUp(features[1],      features[0], features[0],            bilinear, dropout)
        
        
        # 输出层
        self.outc = OutConv(features[0], out_channels)
        
        # 存储注意力权重用于可视化
        self.attention_weights = []
    
    def forward(self, x):
        # 编码路径
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # 解码路径 (带注意力)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Collect attention maps from all decoder stages
        self.attention_weights = [
            self.up1.attention.last_attention_map,
            self.up2.attention.last_attention_map,
            self.up3.attention.last_attention_map,
            self.up4.attention.last_attention_map,
        ]
        
        # 输出
        logits = self.outc(x)
        return logits
    
    def get_attention_maps(self):
        """获取注意力权重图"""
        return self.attention_weights
    
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
            'model_name': 'Attention U-Net',
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

def create_attention_unet_model(config):
    """根据配置创建Attention U-Net模型"""
    model_config = config['model']
    
    model = AttentionUNet(
        in_channels=model_config['in_channels'],
        out_channels=model_config['out_channels'],
        features=model_config['features'],
        dropout=model_config.get('dropout', 0.0)
    )
    
    return model

def visualize_attention_weights(model, input_tensor, save_path=None):
    """可视化注意力权重"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 前向传播获取注意力权重
    model.eval()
    with torch.no_grad():
        _ = model(input_tensor)
        attention_maps = model.get_attention_maps()
    
    if not attention_maps:
        print("No attention weights found. Make sure the model is AttentionUNet.")
        return
    
    # 可视化每层的注意力权重
    num_layers = len(attention_maps)
    fig, axes = plt.subplots(1, num_layers + 1, figsize=(15, 4))
    
    # 显示原始图像
    original_img = input_tensor[0, 0].cpu().numpy()
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 显示每层的注意力权重
    for i, attention_map in enumerate(attention_maps):
        # 取第一个样本和第一个通道
        att_map = attention_map[0, 0].cpu().numpy()
        
        # 上采样到原始图像尺寸
        att_map_resized = np.array(Image.fromarray(att_map).resize(
            (original_img.shape[1], original_img.shape[0])
        ))
        
        axes[i+1].imshow(att_map_resized, cmap='hot', alpha=0.7)
        axes[i+1].imshow(original_img, cmap='gray', alpha=0.3)
        axes[i+1].set_title(f'Attention Layer {i+1}')
        axes[i+1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

# 测试函数
def test_attention_unet():
    """测试Attention U-Net模型"""
    model = AttentionUNet(in_channels=1, out_channels=1)
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
    
    # 比较参数数量
    unet_params = sum(p.numel() for p in AttentionUNet(1, 1).parameters())
    print(f"\nAttention U-Net total parameters: {unet_params:,}")
    
    return model, output

if __name__ == "__main__":
    test_attention_unet()