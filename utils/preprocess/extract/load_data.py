import glob
import os
import sys

import numpy as np
from loguru import logger
from tqdm import tqdm

sys.path.append(os.getcwd())


def load_spec(spec_dir):
    pattern = f"{spec_dir}/*.npy"
    logger.info(f"pattern: {pattern}")
    file_list = glob.glob(pattern)
    file_list.sort()
    y = []

    for idx, file in tqdm(enumerate(file_list), desc=f"load {pattern}..."):
        if idx == 29:
            logger.warning("只读取1993-2021年的数据, 跳过2022年的数据")
            logger.warning(f"跳过2022年的数据: {file}")
            break
        data = np.load(file)
        y.append(data)

    y = np.concatenate(y, axis=0)
    logger.debug(f"y.shape: {y.shape}")  # (84736, 36, 24) (time, freq, direction)
    return y.astype(np.float32)


def load_wind(filepath, skip_2022=True):
    pattern = f"{filepath}/*.npy"
    logger.info(f"pattern: {pattern}")
    file_list = glob.glob(pattern)
    file_list.sort()
    y = []
    for idx, file in tqdm(enumerate(file_list), desc=f"load {pattern}..."):
        data = np.load(file)
        if skip_2022 == True and idx == 29:
            logger.warning("只读取1993-2021年的数据, 跳过2022年的数据")
            logger.warning(f"跳过2022年的数据: {file}")
            break

        y.append(data)

    y = np.concatenate(y, axis=0)

    logger.debug(
        f"wind.shape: {np.asarray(y).shape}"
    )  # (84736, 3) (time, wind(speed,cos,sin))

    return y.astype(np.float32)
