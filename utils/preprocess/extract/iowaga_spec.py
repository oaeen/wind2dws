import glob
import os
import sys
from pathlib import Path

import numpy as np
from loguru import logger
from netCDF4 import Dataset
from tqdm import tqdm

sys.path.append(os.getcwd())

from config import Config
from utils.preprocess.extract.load_data import load_spec


def extract_wave_data(id, data_dir, save_dir, START_YEAR=1993, END_YEAR=2022):
    """
    :param id: 浮标id
    :param filepath: 海浪谱数据存储路径
    :param START_YEAR: 提取数据开始年份
    :param END_YEAR: 提取数据结束年份

    从 START_YEAR 到 END_YEAR,
    读取每年的 netcdf 文件, 提取浮标 id 对应的数据, 保存为 npy 文件
    保存为 npy 文件 (shape: (365天*每天8个时刻, 30个海浪谱频率, 24个方向))
    """
    save_dir = f"{save_dir}/{id}"
    Path.mkdir(Path(save_dir), parents=True, exist_ok=True)

    for year in tqdm(range(START_YEAR, END_YEAR + 1), desc="extracting..."):
        if os.path.exists(f"{save_dir}/{year}.npy"):
            logger.success(f"{year}.npy exists")
            continue

        logger.info(f"\nProcessing year {year}...")
        year_folder = os.path.join(data_dir, str(year))
        # 使用glob模块查找包含id字段的文件
        file_list = glob.glob(os.path.join(year_folder, f"*{id}*"))
        file_list.sort()

        # dilute data
        spec_list = []
        for i, file in enumerate(tqdm(file_list, desc=f"Reading {year}...")):
            logger.info(f"\nProcessing {year}#{i+1}...")
            logger.info(f"Reading {file}...")
            spec = Dataset(file).variables["efth"]
            spec = spec[:, 0, :, :]  # 248 采样点, 36 个频率, 24 个方向
            spec = np.nan_to_num(spec)  # 将 NaN 值替换为 0
            spec = spec.filled(0)  # Convert masked value to 0

            # spec = 10**spec  # Convert ERA5 format to wavespectra format

            spec_list.append(spec)
        spec = np.concatenate(spec_list, axis=0)  # 把一年的数据拼接到一块
        logger.debug(f"Shape of {year} is {spec.shape}")
        np.save(f"{save_dir}/{year}.npy", spec)
        logger.success(f"Save {year}.npy successfully!")


if __name__ == "__main__":
    config = Config()

    # 提取IOWAGA的数据
    save_dir = f"{config.processed_data_dir}/IOWAGA/extract"

    data_dir = f"{config.raw_data_dir}/IOWAGA/SPEC_AT_BUOYS"
    for buyo_id in ["CDIP028", "CDIP045", "46219", "CDIP093", "CDIP107"]:
        extract_wave_data(buyo_id, data_dir, save_dir)

    # data_dir = f"{config.raw_data_dir}/IOWAGA/SPEC_NW139TO100"
    # for loc in [
    #     "W1215N335",
    #     "W1210N330",
    #     "W1205N330",
    #     "W1205N325",
    #     "W1200N320",
    #     "W1195N320",
    #     "W1190N320",
    # ]:
    #     extract_wave_data(loc, data_dir, save_dir)

    for area, pos in [
        ("SW", "W1200S480"),
        ("NE100to139", "E1150N065"),
        ("NW139to100", "W1200N020"),
    ]:
        data_dir = f"{config.raw_data_dir}/IOWAGA/SPEC_{area}"
        extract_wave_data(pos, data_dir, save_dir)
