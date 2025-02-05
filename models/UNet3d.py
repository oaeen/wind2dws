"""Adapted from https://github.com/jphdotam/Unet3D/blob/main/unet3d.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet3d(nn.Module):
    def __init__(self, in_channels, out_channels=16):
        """A simple 3D Unet, adapted from a 2D Unet from https://github.com/milesial/Pytorch-UNet/tree/master/unet
        Arguments:
          in_channels = number of input channels;
          out_channels = number of output channels/classes

        """
        super(UNet3d, self).__init__()
        self.channels = [
            out_channels,  # 0 16
            out_channels * 2,  # 1 32
            out_channels * 4,  # 2 64
            out_channels * 8,  # 3 128
            out_channels * 16,  # 4 256
        ]
        self.convtype = nn.Conv3d

        self.inc = DoubleConv(in_channels, self.channels[0], conv_type=self.convtype)
        self.down1 = Down(self.channels[0], self.channels[1], conv_type=self.convtype)
        self.down2 = Down(self.channels[1], self.channels[2], conv_type=self.convtype)
        self.down3 = Down(self.channels[2], self.channels[3], conv_type=self.convtype)
        factor = 2
        self.down4 = Down(
            self.channels[3], self.channels[4] // factor, conv_type=self.convtype
        )
        self.up1 = Up(self.channels[4], self.channels[3] // factor)
        self.up2 = Up(self.channels[3], self.channels[2] // factor)
        self.up3 = Up(self.channels[2], self.channels[1] // factor)
        self.up4 = Up(self.channels[1], self.channels[0])

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => SiLU) * 2"""

    def __init__(
        self, in_channels, out_channels, conv_type=nn.Conv3d, mid_channels=None
    ):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            conv_type(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.SiLU(inplace=True),
            conv_type(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, conv_type=nn.Conv3d):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2), DoubleConv(in_channels, out_channels, conv_type=conv_type)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # if trilinear, use the normal convolutions to reduce the number of channels
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffZ = x2.size(2) - x1.size(2)  # depth difference
        diffY = x2.size(3) - x1.size(3)  # height difference
        diffX = x2.size(4) - x1.size(4)  # width difference

        x1 = F.pad(
            x1,
            [
                diffX // 2,
                diffX - diffX // 2,  # width padding
                diffY // 2,
                diffY - diffY // 2,  # height padding
                diffZ // 2,
                diffZ - diffZ // 2,  # depth padding
            ],
        )
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
