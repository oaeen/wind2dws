import glob
import os
import sys
from pathlib import Path

import numpy as np
from loguru import logger
from netCDF4 import Dataset
from tqdm import tqdm
import xarray as xr

sys.path.append(os.getcwd())

from config import Config


def extract_wave_data(
    point_name, lat, lon, data_dir, save_dir, START_YEAR=1993, END_YEAR=2022
):
    """
    :param id: 浮标id
    :param filepath: 海浪谱数据存储路径
    :param START_YEAR: 提取数据开始年份
    :param END_YEAR: 提取数据结束年份

    从 START_YEAR 到 END_YEAR,
    读取每年的 netcdf 文件, 提取浮标 id 对应的数据, 保存为 npy 文件
    保存为 npy 文件 (shape: (365天*每天8个时刻, 30个海浪谱频率, 24个方向))
    """
    save_dir = f"{save_dir}/spec_extract_{point_name}"
    Path.mkdir(Path(save_dir), parents=True, exist_ok=True)

    if lon < 0:
        lon = 360 + lon

    for year in tqdm(range(START_YEAR, END_YEAR + 1), desc="extracting..."):
        if os.path.exists(f"{save_dir}/{year}.npy"):
            logger.success(f"{year}.npy exists")
            continue

        logger.info(f"\nProcessing year {year}...")
        year_folder = f"{data_dir}/{year}"
        logger.debug(f"year_folder: {year_folder}")

        # 使用 glob 获取目录下的所有文件列表
        file_list = glob.glob(f"{year_folder}/*.nc")
        file_list.sort()

        # dilute data
        spec_list = []

        for i, file in enumerate(tqdm(file_list, desc=f"Reading {year}...")):
            logger.info(f"\nProcessing {year}#{i+1}...")
            logger.info(f"Reading {file}...")

            spec = xr.open_dataset(file, engine="netcdf4")
            spec = spec["d2fd"]
            spec = spec.sel(latitude=lat, longitude=lon)
            spec = np.asarray(spec.values)
            spec = spec[::3, :, :]
            spec = 10**spec  #  Convert ERA5 format to wavespectra format
            spec = np.nan_to_num(spec)  # 将缺失值填充为0
            spec_list.append(spec)

        spec = np.concatenate(spec_list, axis=0)  # 把一年的数据拼接到一块

        logger.debug(f"Shape of {year} is {spec.shape}")
        np.save(f"{save_dir}/{year}.npy", spec)
        logger.success(f"Save {year}.npy successfully!")


if __name__ == "__main__":
    config = Config()

    # 提取ERA5的海浪谱数据
    data_dir = f"H:/ERA5/Spec_Global/data_5.0D_1H"
    save_dir = f"{config.processed_data_dir}/ERA5/extract"
    era5_points = {
        # "PointA": (5, -120),  # 5°N, 120°W
        # "PointB": (15, 115),  # 15°N, 115°E
        "PointC": (-50, 165),  # 50°S, 165°E
    }
    for point_name, (lat, lon) in era5_points.items():
        extract_wave_data(point_name, lat, lon, data_dir, save_dir)
