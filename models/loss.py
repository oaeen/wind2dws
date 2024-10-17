import torch
import torch.nn as nn
import numpy as np

import sys
import os


class AdaptiveMSELoss(nn.Module):
    def __init__(self, epsilon=1e-2):
        """
        参数：
        - epsilon: 平滑项，避免除以零
        """
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        """
        前向传播，计算损失值。

        参数：
        - y_pred: 模型的预测值，形状为 (batch_size, 30, 24)
        - y_true: 真实值，形状为 (batch_size, 30, 24)

        返回：
        - loss: 计算得到的总损失值
        """

        weight = torch.clamp(y_true, min=1e-3, max=0.5)
        loss = torch.mean((y_pred - y_true) ** 2 / weight)

        return loss


if __name__ == "__main__":
    y_pred = torch.rand(2, 30, 24).to("cuda")
    y_true = torch.rand(2, 30, 24).to("cuda")

    loss = AdaptiveMSELoss()
    print(loss(y_pred, y_true))
