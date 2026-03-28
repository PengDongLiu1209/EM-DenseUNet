import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class DenseLayer(nn.Module):
    """Dense Layer: BN-ReLU-Conv(1x1)-BN-ReLU-Conv(3x3)"""
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                          kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        # Apply the sequence of operations manually
        out = self.norm1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu2(out)
        new_features = self.conv2(out)
        
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

class DenseBlock(nn.Module):
    """Dense Block consisting of multiple Dense Layers"""
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, x):
        for name, layer in self.named_children():
            x = layer(x)
        return x

class Transition(nn.Module):
    """Transition Layer: BN-ReLU-Conv(1x1)-AvgPool(2x2)"""
    def __init__(self, num_input_features, num_output_features):
        super(Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        # Apply the sequence of operations manually
        out = self.norm(x)
        out = self.relu(out)
        out = self.conv(out)
        out = self.pool(out)
        return out

class DenseNet(nn.Module):
    """DenseNet-121 model class for segmentation tasks (标准DenseNet-121架构)
    
    基于原始DenseNet-121论文实现，专门用于分割任务，使用标准配置：
    - growth_rate=32
    - block_config=(6, 12, 24, 16) 
    - num_init_features=64
    - bn_size=4
    - input_channels=3 (固定为RGB图像)
    
    Args:
        n_classes (int): number of segmentation classes 
                        - 1: binary segmentation (background, object)
                        - 3: your case (background, fruit, stem)
                        - N: multi-class segmentation
    """

    def __init__(self, n_classes):

        super(DenseNet, self).__init__()
        
        # 固定的DenseNet-121标准配置
        growth_rate = 32
        block_config = (6, 12, 24, 16)
        num_init_features = 64
        bn_size = 4
        drop_rate = 0
        input_channels = 3  # 固定为RGB图像
        
        # 专门用于分割任务
        self.n_classes = n_classes

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(input_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        self.skip_channels = [num_init_features]  # 用于分割任务的跳跃连接
        
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers=num_layers, num_input_features=num_features,
                              bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            
            self.skip_channels.append(num_features)  # 保存每个dense block的输出通道数
            
            if i != len(block_config) - 1:
                trans = Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # 分割任务：解码器
        self._init_segmentation_decoder(num_features)
            
        # 激活函数（用于分割）
        if n_classes == 1:
            self.activation = nn.Sigmoid()  # 二值分割
        else:
            self.activation = None  # 多类分割使用softmax或其他

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    
    def _init_segmentation_decoder(self, num_features):
        """初始化分割任务的解码器"""
        # 解码器部分 (新增用于分割)
        self.up_conv4 = nn.ConvTranspose2d(num_features, 512, kernel_size=2, stride=2)
        self.up_block4 = self._make_up_block(512 + 1024, 256)  # 1024是denseblock4的输出
        
        self.up_conv3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.up_block3 = self._make_up_block(256 + 512, 128)   # 512是denseblock3的输出
        
        self.up_conv2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.up_block2 = self._make_up_block(128 + 256, 64)    # 256是denseblock2的输出
        
        self.up_conv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.up_block1 = self._make_up_block(64 + 64, 32)      # 64是初始conv的输出
        
        # 最终分类层
        self.final_conv = nn.Conv2d(32, self.n_classes, kernel_size=1)
    
    def _make_up_block(self, in_channels, out_channels):
        """创建上采样后的卷积块"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """分割任务的前向传播"""
        return self._forward_segmentation(x)
    
    def _forward_segmentation(self, x):
        """分割任务的前向传播"""
        # 保存输入尺寸
        input_size = x.size()[2:]
        
        # ===== 编码器 (与原始DenseNet-121完全一致的特征提取) =====
        skip_features = []
        
        # 逐层提取特征并保存跳跃连接
        for name, layer in self.features.named_children():
            if name == 'pool0':
                # 在pool0之前保存conv0的输出
                skip_features.append(x)
            
            x = layer(x)
            
            # 保存dense block的输出用于跳跃连接
            if 'denseblock' in name:
                skip_features.append(x)
        
        # 最终特征处理
        x = F.relu(x, inplace=True)
        
        # ===== 解码器 =====
        # 逐步上采样并融合跳跃连接
        # skip_features[0]: conv0输出 (64通道)
        # skip_features[1]: denseblock1输出 (256通道)  
        # skip_features[2]: denseblock2输出 (512通道)
        # skip_features[3]: denseblock3输出 (1024通道)
        # 注意：denseblock4的输出直接作为解码器的输入，不需要保存
        
        x = self.up_conv4(x)    # 上采样到1/16分辨率
        x = torch.cat([x, skip_features[3]], dim=1)  # 融合denseblock3
        x = self.up_block4(x)
        
        x = self.up_conv3(x)    # 上采样到1/8分辨率
        x = torch.cat([x, skip_features[2]], dim=1)  # 融合denseblock2
        x = self.up_block3(x)
        
        x = self.up_conv2(x)    # 上采样到1/4分辨率
        x = torch.cat([x, skip_features[1]], dim=1)  # 融合denseblock1
        x = self.up_block2(x)
        
        x = self.up_conv1(x)    # 上采样到1/2分辨率
        x = torch.cat([x, skip_features[0]], dim=1)  # 融合conv0输出
        x = self.up_block1(x)
        
        # 最终分类
        x = self.final_conv(x)
        
        # 确保输出尺寸与输入匹配
        if x.size()[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        
        # 应用激活函数
        if self.activation is not None:
            x = self.activation(x)
            
        return x

if __name__ == "__main__":
    # Test DenseNet model for segmentation
    print("Testing DenseNet-121 Segmentation Model...")
    
    # 创建DenseNet-121分割模型 (与DenseUNet调用方式完全一致)
    print("\n1. 二值分割测试 (背景 vs 目标):")
    binary_model = DenseNet(n_classes=1)  # 二值分割
    binary_model.eval()
    
    # 创建您的3类分割模型 (背景, 果实, 茎)
    print("\n2. 三类分割测试 (背景, 果实, 茎):")
    fruit_model = DenseNet(n_classes=3)  # 您的场景：背景, 果实, 茎
    fruit_model.eval()
    
    # 测试不同输入尺寸
    test_sizes = [(224, 224), (256, 256), (512, 512)]
    
    print("\n测试不同输入尺寸:")
    for size in test_sizes:
        test_input = torch.randn(1, 3, size[0], size[1])
        
        with torch.no_grad():
            # 二值分割测试
            binary_output = binary_model(test_input)
            print(f"二值分割 - Input: {test_input.shape} -> Output: {binary_output.shape}")
            assert binary_output.shape[2:] == test_input.shape[2:], f"Output size mismatch"
            
            # 三类分割测试
            fruit_output = fruit_model(test_input)
            print(f"三类分割 - Input: {test_input.shape} -> Output: {fruit_output.shape}")
            assert fruit_output.shape[2:] == test_input.shape[2:], f"Output size mismatch"
    
    # 计算模型参数
    binary_params = sum(p.numel() for p in binary_model.parameters())
    fruit_params = sum(p.numel() for p in fruit_model.parameters())
    
    print(f"\n模型统计信息:")
    print(f"二值分割模型参数: {binary_params:,} ({binary_params * 4 / 1024 / 1024:.2f} MB)")
    print(f"三类分割模型参数: {fruit_params:,} ({fruit_params * 4 / 1024 / 1024:.2f} MB)")
    
    print("\n" + "="*60)
    print("DenseNet-121 分割模型特性:")
    print("="*60)
    print("✓ 标准DenseNet-121架构 (6,12,24,16层配置)")
    print("✓ 完全基于原始DenseNet论文实现")
    print("✓ 专门用于分割任务，最贴近原始DenseNet")
    print("✓ 与DenseUNet完全相同的调用方式:")
    print("  - 二值分割: DenseNet(n_classes=1)")
    print("  - 您的场景: DenseNet(n_classes=3)  # 背景,果实,茎")
    print("  - 多类分割: DenseNet(n_classes=N)")
    print("✓ 高效的密集连接和特征重用")
    print("✓ 固定RGB输入 (3通道)")
    print(f"✓ 约{binary_params/1e6:.2f}M参数 (二值), {fruit_params/1e6:.2f}M参数 (三类)")
    print("="*60)