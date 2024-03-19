import torch
import torch.nn as nn
import torch.nn.functional as F

class OutConv(nn.Module):
    """
        Output Convolutional Block
        Args:
            in_channels: input channels
            out_channels: output channels
        Returns:
            Output Convolutional Block
    """ 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1)

    def forward(self, x):
        return self.conv(x)

class Up(nn.Module):
    """
        Upsample Block
        Args:
            n: number of convolutions blocks
            in_channels: input channels
            out_channels: output channels
            mode: upsampling mode
            norm_layer: normalization layer
        Returns:
            Upsample Block
    """
    def __init__(self, n, in_channels, out_channels, mode = 'nearest',norm_layer = nn.BatchNorm2d):
        super().__init__()
        if n not in [2, 3]:
            raise ValueError("n must be 2 or 3")
        self.up = nn.Upsample(scale_factor = 2, mode = mode)
        self.conv = DoubleConvBlock(in_channels = in_channels, out_channels = out_channels, norm_layer = norm_layer) if n == 2 else TripleConvBlock(in_channels = in_channels, out_channels = out_channels, norm_layer = norm_layer)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim = 1)
        return self.conv(x)

class Down(nn.Module):
    """
        Downsample Block
        Args:
            n: number of convolutions blocks
            in_channels: input channels
            out_channels: output channels
            norm_layer: normalization layer
        Returns:
            Downsample Block
    """
    def __init__(self, n, in_channels, out_channels, norm_layer = nn.BatchNorm2d):
        super().__init__()
        if n not in [2, 3]:
            raise ValueError("n must be 2 or 3")
        self.block = nn.Sequential(
            nn.MaxPool2d(2, 2),
            DoubleConvBlock(in_channels = in_channels, out_channels = out_channels, norm_layer = norm_layer) if n == 2 else TripleConvBlock(in_channels = in_channels, out_channels = out_channels, norm_layer = norm_layer),
        )
    
    def forward(self, x):
        return self.block(x)

class TripleConvBlock(nn.Module):
    """
        Triple Convolutional Block
        Args:
            in_channels: input channels
            out_channels: output channels
            norm_layer: normalization layer
        Returns:
            TripleConvBlock
    """
    def __init__(self, in_channels, out_channels, norm_layer = nn.BatchNorm2d):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_channels = in_channels, out_channels = out_channels, norm_layer = norm_layer),
            ConvBlock(in_channels = out_channels, out_channels = out_channels, norm_layer = norm_layer),
            ConvBlock(in_channels = out_channels, out_channels = out_channels, norm_layer = norm_layer)
        )    
    
    def forward(self, x):
        return self.block(x)

class DoubleConvBlock(nn.Module):
    """
        Double Convolutional Block
        Args:
            in_channels: input channels
            out_channels: output channels
            norm_layer: normalization layer
        Returns:
            DoubleConvBlock
    """
    def __init__(self, in_channels, out_channels, norm_layer = nn.BatchNorm2d):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_channels = in_channels, out_channels = out_channels, norm_layer = norm_layer),
            ConvBlock(in_channels = out_channels, out_channels = out_channels, norm_layer = norm_layer)
        )    
    
    def forward(self, x):
        return self.block(x)
    
class ConvBlock(nn.Module):
    """
        Convolutional Block
        Args:
            in_channels: input channels
            out_channels: output channels
            norm_layer: normalization layer
        Returns:
            ConvBlock
    """
    def __init__(self, in_channels, out_channels, norm_layer = nn.BatchNorm2d):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            norm_layer(num_features = out_channels)
        )

    def forward(self, x):
        return self.block(x)