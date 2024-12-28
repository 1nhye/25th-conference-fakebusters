import torch
import torch.nn as nn
import torch.nn.functional as F

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, stride, kernel_size):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.use_residual = self.stride == 1 and in_channels == out_channels
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        layers = []
        if self.expand:
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.SiLU())
        layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.block(x)
        else:
            return self.block(x)
        
class EfficientNetB3(nn.Module):
    def __init__(self, w):
        super(EfficientNetB3, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 40, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(40),
            nn.SiLU()
        )
        self.blocks = nn.Sequential(
            MBConvBlock(40, 24, expand_ratio=1, stride=1, kernel_size=3),
            MBConvBlock(24, 24, expand_ratio=1, stride=1, kernel_size=3),
            MBConvBlock(24, 48, expand_ratio=6, stride=2, kernel_size=3),
            MBConvBlock(48, 48, expand_ratio=6, stride=1, kernel_size=3),
            MBConvBlock(48, 80, expand_ratio=6, stride=2, kernel_size=5),
            MBConvBlock(80, 80, expand_ratio=6, stride=1, kernel_size=5),
            MBConvBlock(80, 112, expand_ratio=6, stride=1, kernel_size=5),
            MBConvBlock(112, 112, expand_ratio=6, stride=1, kernel_size=5),
            MBConvBlock(112, 192, expand_ratio=6, stride=2, kernel_size=3),
            MBConvBlock(192, 192, expand_ratio=6, stride=1, kernel_size=3),
            MBConvBlock(192, 192, expand_ratio=6, stride=1, kernel_size=3),
            MBConvBlock(192, 320, expand_ratio=6, stride=1, kernel_size=5)
        )

        feature_map_width = w // (2 ** 5)  # Downsampling through 5 pooling layers
        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, feature_map_width)),
            nn.Flatten(),
            nn.Linear(1280 * feature_map_width, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x
