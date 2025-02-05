import os
import sys
from math import e
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm

sys.path.append(os.getcwd())
import gc

from config import Config
from utils.preprocess.extract.load_data import load_spec, load_wind
from utils.preprocess.prepare.scale_spec import *


def prepare_wind(wind_dir, save_dir, scale=False):
    """
    加载每年的数据, 对海浪谱进行 scale, 并分别保存为训练集和测试集
    """

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    wind = load_wind(wind_dir)
    u_wind = wind[:, 0, :, :]
    v_wind = wind[:, 1, :, :]

    u_wind = u_wind**2 * np.sign(u_wind)
    v_wind = v_wind**2 * np.sign(v_wind)

    wind = np.stack([u_wind, v_wind], axis=1)
    print(f"wind.shape: {wind.shape}")
    wind_train = wind[: int(len(wind) * 0.8)]
    wind_test = wind[int(len(wind) * 0.8) :]

    logger.debug(f"wind_train.shape: {wind_train.shape}")
    logger.debug(f"wind_test.shape: {wind_test.shape}")

    np.save(f"{save_dir}/train.npy", wind_train.astype(np.float32))
    np.save(f"{save_dir}/test.npy", wind_test.astype(np.float32))


if __name__ == "__main__":
    config = Config()

    era5_points = {
        # "PointA": (5, -120),  # 5°N, 120°W
        # "PointB": (15, 115),  # 15°N, 115°E
        "PointC": (-50, 165),  # 50°S, 165°E
        # "CDIP028": (34, -118.5),  # 34°N, 118.5°W aka PointD
    }
    for location, value in era5_points.items():
        wind_dir = f"{config.processed_data_dir}/ERA5/extract/wind_extracted_{location}"
        save_dir = f"{config.processed_data_dir}/ERA5/input/wind_input_{location}"
        prepare_wind(wind_dir, save_dir, False)

    # location = "large"
    # wind_dir = f"{config.processed_data_dir}/ERA5/extract/wind_extracted_{location}"
    # save_dir = f"{config.processed_data_dir}/ERA5/input/wind_input_{location}"
    # prepare_wind(wind_dir, save_dir, False)

    # location = "large_PB"
    # wind_dir = f"{config.processed_data_dir}/ERA5/extract/wind_extracted_{location}"
    # save_dir = f"{config.processed_data_dir}/ERA5/input/wind_input_{location}"
    # prepare_wind(wind_dir, save_dir, False)
