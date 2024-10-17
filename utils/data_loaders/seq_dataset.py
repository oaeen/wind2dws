import numpy as np
import torch


class Seq_Dataset(torch.utils.data.Dataset):
    """
    生成一个数据集，用于训练模型
    返回的数据格式为单个样本：(输入海浪谱数据, 风场数据, 预测目标海浪谱数据)
    """

    def __init__(
        self,
        global_wind,
        local_wind,
        target_data,
        global_window_size=8 * 10,
        local_window_size=8 * 2,
        steps=17,
    ):

        self.pacific_wind = global_wind
        self.coastal_wind = local_wind
        self.target_data = target_data
        self.pacific_window_size = global_window_size
        self.coastal_window_size = local_window_size
        self.steps = steps
        self.sample_size = (len(global_wind) - global_window_size) // self.steps + 1

    def __getitem__(self, idx):
        """
        Get a single sample with shape (window_size, longitude, latitude, feature)

        :param idx: The index of the sample to retrieve
        :return: A tuple containing the input and target data for the sample
        """
        idx = idx * self.steps
        end_idx = idx + self.pacific_window_size
        pacific_wind = self.pacific_wind[idx:end_idx]

        coastal_start_idx = end_idx - self.coastal_window_size
        coastal_wind = self.coastal_wind[coastal_start_idx:end_idx]

        target = self.target_data[end_idx - 1]

        return pacific_wind, coastal_wind, target

    def __len__(self):
        """
        Calculate the number of samples in the dataset
        """
        return self.sample_size
