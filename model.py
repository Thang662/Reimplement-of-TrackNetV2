from model_parts import *
from torchinfo import summary

class TrackNetV2(nn.Module):
    """
        TrackNetV2
        Args:
            in_channels: input channels
            out_channels: output channels
            norm_layer: normalization layer
        Returns:
            TrackNetV2
    """
    def __init__(self, in_channels, out_channels, norm_layer = nn.BatchNorm2d):
        super().__init__()
        self.conv = DoubleConvBlock(in_channels = in_channels, out_channels = 64, norm_layer = norm_layer)
        self.down1 = Down(n = 2, in_channels = 64, out_channels = 128, norm_layer = norm_layer)
        self.down2 = Down(n = 3, in_channels = 128, out_channels = 256, norm_layer = norm_layer)
        self.down3 = Down(n = 3, in_channels = 256, out_channels = 512, norm_layer = norm_layer)
        self.up1 = Up(n = 3, in_channels = 768, out_channels = 256, norm_layer = norm_layer)
        self.up2 = Up(n = 2, in_channels = 384, out_channels = 128, norm_layer = norm_layer)
        self.up3 = Up(n = 2, in_channels = 192, out_channels = 64, norm_layer = norm_layer)
        self.out = OutConv(in_channels = 64, out_channels = out_channels)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.down3(x3)
        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.out(x)
    
if __name__ == "__main__":
    model = TrackNetV2(in_channels = 9, out_channels = 3)
    summary(model, input_size = (4, 9, 512, 288), device = "cpu", col_names = ('input_size', 'output_size', 'num_params', 'kernel_size', 'mult_adds'), verbose = 1)