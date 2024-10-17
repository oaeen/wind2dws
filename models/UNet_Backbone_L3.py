import torch
from torch import nn

from models.layers import CBAM, DoubleConv, Down, Up, OutConv


class UNet_Backbone_L3(nn.Module):
    def __init__(
        self,
        input_channels=1,
        extract_channels=128,
        reduction_ratio=16,
    ):
        super(UNet_Backbone_L3, self).__init__()
        self.n_channels = input_channels

        c1 = extract_channels  # 64
        c2 = c1 * 2  # 128
        c3 = c2 * 2  # 256
        c4 = c3 * 2  # 512

        self.inc = DoubleConv(self.n_channels, c1)
        self.cbam1 = CBAM(c1, reduction_ratio=reduction_ratio)
        self.down1 = Down(c1, c2)
        self.cbam2 = CBAM(c2, reduction_ratio=reduction_ratio)
        self.down2 = Down(c2, c3)
        self.cbam3 = CBAM(c3, reduction_ratio=reduction_ratio)
        self.down3 = Down(c3, c4)
        self.cbam4 = CBAM(c4, reduction_ratio=reduction_ratio)

        self.up1 = Up(c4, c3)
        self.up2 = Up(c3, c2)
        self.up3 = Up(c2, c1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x1Att = self.cbam1(x1)
        x2Att = self.cbam2(x2)
        x3Att = self.cbam3(x3)
        x4Att = self.cbam4(x4)

        x = self.up1(x4Att, x3Att)
        x = self.up2(x, x2Att)
        x = self.up3(x, x1Att)

        return x
