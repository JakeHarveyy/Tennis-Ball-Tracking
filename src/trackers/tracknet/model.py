import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
import collections
from tqdm import tqdm
import pandas as pd

class ConvBlock(nn.Module):
    """
    Basic convolutional block with Conv2D + ReLU + BatchNorm.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Convolution kernel size (default: 3)
        pad (int): Padding size (default: 1)
        stride (int): Convolution stride (default: 1)
        bias (bool): Whether to use bias in convolution (default: True)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, pad=1, stride=1, bias=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.block(x)

class TrackNet(nn.Module):
    """
    TrackNet model for tennis ball tracking using encoder-decoder architecture.
    
    Takes 3 consecutive frames (9 channels) as input and outputs a 256-class
    heatmap for ball localization. Architecture follows VGG16-style encoder
    with DeconvNet-style decoder for pixel-level classification.
    
    Args:
        out_channels (int): Number of output classes (default: 256)
    """
    def __init__(self, out_channels=256):
        super().__init__()
        self.out_channels = out_channels

        # Encoder: VGG16-style feature extraction
        # Block 1: 9 -> 64 channels, spatial size: 640x360 -> 320x180
        self.conv1 = ConvBlock(in_channels=9, out_channels=64)
        self.conv2 = ConvBlock(in_channels=64, out_channels=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: 64 -> 128 channels, spatial size: 320x180 -> 160x90
        self.conv3 = ConvBlock(in_channels=64, out_channels=128)
        self.conv4 = ConvBlock(in_channels=128, out_channels=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3: 128 -> 256 channels, spatial size: 160x90 -> 80x45
        self.conv5 = ConvBlock(in_channels=128, out_channels=256)
        self.conv6 = ConvBlock(in_channels=256, out_channels=256)
        self.conv7 = ConvBlock(in_channels=256, out_channels=256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4: 256 -> 512 channels (bottleneck), spatial size: 80x45
        self.conv8 = ConvBlock(in_channels=256, out_channels=512)
        self.conv9 = ConvBlock(in_channels=512, out_channels=512)
        self.conv10 = ConvBlock(in_channels=512, out_channels=512)

        # Decoder: DeconvNet-style upsampling
        # Upsample 1: 80x45 -> 160x90, 512 -> 256 channels
        self.ups1 = nn.Upsample(scale_factor=2)
        self.conv11 = ConvBlock(in_channels=512, out_channels=256)
        self.conv12 = ConvBlock(in_channels=256, out_channels=256)
        self.conv13 = ConvBlock(in_channels=256, out_channels=256)

        # Upsample 2: 160x90 -> 320x180, 256 -> 128 channels
        self.ups2 = nn.Upsample(scale_factor=2)
        self.conv14 = ConvBlock(in_channels=256, out_channels=128)
        self.conv15 = ConvBlock(in_channels=128, out_channels=128)

        # Upsample 3: 320x180 -> 640x360, 128 -> 64 -> out_channels
        self.ups3 = nn.Upsample(scale_factor=2)
        self.conv16 = ConvBlock(in_channels=128, out_channels=64)
        self.conv17 = ConvBlock(in_channels=64, out_channels=64)
        self.conv18 = ConvBlock(in_channels=64, out_channels=self.out_channels)

        self.softmax = nn.Softmax(dim=1)
        self._init_weights()
                  
    def forward(self, x):
        """
        Forward pass through TrackNet.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, 9, H, W)
            
        Returns:
            torch.Tensor: Output heatmap of shape (N, 256, H, W)
        """
        # Encoder path
        x = self.conv1(x)
        x = self.conv2(x)    
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        
        # Decoder path
        x = self.ups1(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.ups2(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.ups3(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)

        return x
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.uniform_(module.weight, -0.05, 0.05)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)    
    


