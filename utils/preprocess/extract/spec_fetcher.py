import os
import sys
import time
from datetime import datetime, timedelta

import netCDF4
import numpy as np
import torch
from loguru import logger

sys.path.append(os.getcwd())

from config import Config


def get_iowaga_spec_time_list(spec, config=Config()):
    """
    从提取的IOWAGA谱测试集数据中获取时间戳
    """
    # 测试集为 29年 * 0.2 = 5.8年的数据，取后5年的数据 ["2017", "2018", "2019", "2020", "2021"]
    # 后5年的数据为 5 * 365 * 8 + 8(闰一天) = 14608 个样本
    test_start_idx = 14608

    start_time = datetime(2017, 1, 1, 0, 0, 0)
    iowaga_time_list = [
        start_time + timedelta(hours=3 * idx) for idx in range(test_start_idx)
    ]
    return iowaga_time_list


def get_target_year_spec(spec, config=Config()):
    """
    获取指定年份的样本谱图
    """
    # 筛选出 2020年 或者 2015年的数据
    target_year = 2020

    logger.debug(f"绘制IOWAGA数据的指定年份样本谱图和平均谱图")
    time_list = get_iowaga_spec_time_list(config)
    # 从后往前取5年的数据(14608个样本)，同时也规避了实验输入为时间窗口大小的影响
    test_start_idx = 14608
    # 从后往前取5年的数据(14608个样本)
    spec = spec[-test_start_idx:]

    spec_target_year = []
    # 因为输入的是滑动窗口的数据，所以需要减去滑动窗口的大小
    for idx, time in enumerate(time_list):
        if time.year == target_year:
            spec_target_year.append(spec[idx])

    spec_target_year = np.asarray(spec_target_year)
    logger.info(f"spec_target_year{target_year}.shape: {spec_target_year.shape}")

    return spec_target_year


def get_specific_spec_samples(spec, target_year, month_indices, config=Config()):

    logger.debug(f"绘制IOWAGA数据的指定年份样本谱图和平均谱图")
    time_list = get_iowaga_spec_time_list(config)
    # 从后往前取5年的数据(14608个样本)，同时也规避了实验输入为时间窗口大小的影响
    test_start_idx = 14608
    # 从后往前取5年的数据(14608个样本)
    spec = spec[-test_start_idx:]

    spec_target_year = []
    spec_samples = []
    # 因为输入的是滑动窗口的数据，所以需要减去滑动窗口的大小
    for idx, time in enumerate(time_list):
        if time.year == target_year:
            spec_target_year.append(spec[idx])
            # 查找接近 1/4/7/10 月 1日 12时0分0秒 的样本
            if (
                time.month in month_indices
                and time.day == 1
                and np.abs((time.hour + time.minute / 60) - 12) <= 1
            ):  # 指定月份1日的12点左右
                logger.debug(
                    f"找到指定 {config.y_data_source}: {config.y_location}#{time} 的样本"
                )
                spec_samples.append(spec[idx])

    spec_avg = np.mean(spec_target_year, axis=0)
    spec_samples = np.array(spec_samples)
    return spec_avg, spec_samples


def get_season_avg_spec(spec, target_year, config=Config()):

    logger.debug(f"绘制IOWAGA数据的四季图")
    time_list = get_iowaga_spec_time_list(spec, config)

    # specific for CDIP028
    # 分别从中取 指定年份数据 中 春夏秋冬的数据 的 index
    spring_index = []
    summer_index = []
    autumn_index = []
    winter_index = []

    for idx, time in enumerate(time_list):
        if time.year == target_year:
            if time.month in [3, 4, 5]:
                spring_index.append(idx)
            elif time.month in [6, 7, 8]:
                summer_index.append(idx)
            elif time.month in [9, 10, 11]:
                autumn_index.append(idx)
            elif time.month in [12, 1, 2]:
                winter_index.append(idx)

    spring_spec = np.mean(spec[spring_index], axis=0)
    summer_spec = np.mean(spec[summer_index], axis=0)
    autumn_spec = np.mean(spec[autumn_index], axis=0)
    winter_spec = np.mean(spec[winter_index], axis=0)

    return [winter_spec, spring_spec, summer_spec, autumn_spec]
