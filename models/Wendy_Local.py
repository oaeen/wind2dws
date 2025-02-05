import sys
import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchinfo

sys.path.append(os.getcwd())
from models.layers import Wind2Spec
from models.model import calculate_model_size
from models.UNet3d import UNet3d


class Wendy_Local(nn.Module):
    def __init__(
        self, freq_num=36, large_days=10, local_days=3, unet3d_out=16, out_in=25
    ):
        super(Wendy_Local, self).__init__()

        self.local_encoder = UNet3d(in_channels=2, out_channels=unet3d_out)
        self.local_weight = nn.Parameter(torch.ones(1, 1, local_days * 8, 1, 1))
        self.local_decoder = Wind2Spec(16, freq_num)
        self.out = nn.Conv2d(25, 1, kernel_size=1)

    def forward(self, large, local):

        local = local.permute(0, 2, 1, 3, 4)
        local = self.local_encoder(local)
        local = local * self.local_weight
        local = torch.sum(local, dim=2)
        local = self.local_decoder(local)

        x = self.out(local)

        x = torch.squeeze(x)

        x = torch.nan_to_num(x, -1)

        return x


if __name__ == "__main__":
    batch_size = 2
    large_day = 10
    large_time_windows = 8 * large_day
    large_wind = torch.ones(4, 8 * large_day, 2, 72, 60, dtype=torch.float32).to("cuda")

    local_day = 3
    local_time_windows = 8 * local_day
    local_wind = torch.ones(4, 8 * local_day, 2, 81, 81, dtype=torch.float32).to("cuda")
    model = Wendy_Local().to("cuda")

    output = model(large_wind, local_wind)
    torchinfo.summary(model, input_size=[large_wind.shape, local_wind.shape], depth=1)
    print(output.shape)
    print(f"model size: {calculate_model_size(model)} MB")
