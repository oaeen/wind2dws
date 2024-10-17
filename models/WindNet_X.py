import torch
from torch import nn

import sys
import os

sys.path.append(os.getcwd())

from models.layers import OutConv, DoubleConv, ConvBlock
from models.UNet_Backbone_L3 import UNet_Backbone_L3 as UNet
from models.model import calculate_model_size


class WindNet_X(nn.Module):
    def __init__(self, input_channels=2, output_size=1):
        super(WindNet_X, self).__init__()
        self.g_shared_conv0 = ConvBlock(16, 32)
        self.g_shared_unet = UNet(input_channels=32, extract_channels=64)

        self.l_shared_conv0 = ConvBlock(16, 32)
        self.l_shared_unet = UNet(input_channels=32, extract_channels=64)

        self.g_heads = nn.ModuleList([Wind_Feature_Head(64, 2) for _ in range(10)])
        self.l_heads = nn.ModuleList([Wind_Feature_Head(64, 4) for _ in range(2)])
        self.out_conv = nn.Conv2d(12, 1, kernel_size=1)

    def forward(self, g, l):
        batch_size, g_time, g_channels, g_lon, g_lat = g.shape
        g_sample_days = g_time // 8
        g = g.view(batch_size, g_sample_days, 8, g_channels, g_lon, g_lat)
        g = g.view(batch_size, g_sample_days, 8 * g_channels, g_lon, g_lat)
        outputs = []

        for i in range(g_sample_days):
            g_i = g[:, i, :, :, :]
            g_i = self.g_shared_conv0(g_i)
            g_i = self.g_shared_unet(g_i)
            output = self.g_heads[i](g_i)
            outputs.append(output)

        batch_size, l_time, l_channels, l_lon, l_lat = l.shape
        l_sample_days = l_time // 8
        l = l.view(batch_size, l_sample_days, 8, l_channels, l_lon, l_lat)
        l = l.view(batch_size, l_sample_days, 8 * l_channels, l_lon, l_lat)
        for i in range(2):
            l_i = l[:, i, :, :, :]
            l_i = self.l_shared_conv0(l_i)
            l_i = self.l_shared_unet(l_i)
            output = self.l_heads[i](l_i)
            outputs.append(output)

        x = torch.stack(outputs, dim=0)

        x = x.permute(1, 0, 2)
        x = x.view(batch_size, g_sample_days + l_sample_days, 36, 24)
        x = self.out_conv(x)
        x = torch.squeeze(x)
        x = torch.where(torch.isnan(x), torch.ones_like(x) * -1, x)

        return x


class Wind_Feature_Head(nn.Module):
    def __init__(self, input_channels=16, output_conv_in=8):
        super(Wind_Feature_Head, self).__init__()
        self.out_conv_in = output_conv_in
        self.conv1 = ConvBlock(input_channels, 128)
        self.conv2 = ConvBlock(128, 256)
        self.conv3 = ConvBlock(256, 512)
        self.conv4 = ConvBlock(512, 864)
        self.out_conv = nn.Conv1d(output_conv_in, 1, kernel_size=1)

    def forward(self, x):
        batch_size, channels, _lon, _lat = x.shape
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(batch_size, 864, self.out_conv_in)
        x = x.permute(0, 2, 1)
        x = self.out_conv(x)
        x = torch.squeeze(x)

        return x


if __name__ == "__main__":
    global_time_windows = 8 * 10  # 10 days
    global_channels = global_time_windows
    global_wind = torch.ones(4, 80, 2, 72, 60, dtype=torch.float32).to("cuda")
    local_wind = torch.ones(4, 16, 2, 81, 81, dtype=torch.float32).to("cuda")
    model = WindNet_X().to("cuda")
    output = model(global_wind, local_wind)
    print(f"output shape: {output.shape}")
    print(f"model size: {calculate_model_size(model)} MB")
