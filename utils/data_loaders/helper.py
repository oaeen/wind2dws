import os
import sys

import numpy as np
from loguru import logger
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())

from config import Config
from utils.data_loaders.seq_dataset import Seq_Dataset


def get_dataloader(config=Config()):
    """
    按照batch_size & window_size划分数据集, 并加载到iterableDataset中
    返回train_dataloader, test_dataloader
    """

    train_dataset, test_dataset = get_dataset(config=config)
    logger.debug(
        f"train/test dataset samples: {len(train_dataset)}/{len(test_dataset)}"
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False
    )

    return train_dataloader, test_dataloader


def get_dataset(config=Config()):
    logger.info(f"loading data...")
    X_global_train, X_global_test = load_X_global_wind(config)
    X_local_train, X_local_test = load_X_local_wind(config)

    y_train, y_test = load_y_spec(config)
    train_dataset = Seq_Dataset(
        X_global_train,
        X_local_train,
        y_train,
        config.global_wind_window,
        config.local_wind_window,
        steps=config.train_steps,
    )
    test_dataset = Seq_Dataset(
        X_global_test,
        X_local_test,
        y_test,
        config.global_wind_window,
        config.local_wind_window,
        steps=config.test_steps,
    )

    return train_dataset, test_dataset


def load_X_global_wind(config=Config()):

    wind_dir = f"{config.processed_data_dir}/ERA5/input/wind_input_{config.global_wind_location}"
    logger.success(f"load global wind data from {wind_dir}")

    X_wind_train = np.load(f"{wind_dir}/train.npy")
    X_wind_test = np.load(f"{wind_dir}/test.npy")

    logger.debug(
        f"X_wind_train: {X_wind_train.shape}, X_wind_test: {X_wind_test.shape} "
    )
    return X_wind_train, X_wind_test


def load_X_local_wind(config=Config()):
    wind_dir = f"{config.processed_data_dir}/ERA5/input/wind_input_{config.local_wind_location}"
    logger.success(f"load local wind data from {wind_dir}")

    X_wind_train = np.load(f"{wind_dir}/train.npy")
    X_wind_test = np.load(f"{wind_dir}/test.npy")

    logger.debug(
        f"X_wind_train: {X_wind_train.shape}, X_wind_test: {X_wind_test.shape} "
    )
    return X_wind_train, X_wind_test


def load_y_spec(config=Config()):
    y_train = np.load(f"{config.get_y_data_dir()}/train.npy")
    y_test = np.load(f"{config.get_y_data_dir()}/test.npy")

    logger.success(f"load y data from {config.get_y_data_dir()}")
    logger.debug(f"y_spec_train: {y_train.shape}, y_spec_test: {y_test.shape} ")

    return y_train, y_test


if __name__ == "__main__":
    import sys

    sys.path.append(os.getcwd())
    from config import Config

    config = Config()
    train_dataloader, test_dataloader = get_dataloader(config)
    logger.debug(f"train_dataloader: {len(train_dataloader)}")
    logger.debug(f"test_dataloader: {len(test_dataloader)}")
    for i, (x_spec, y) in enumerate(train_dataloader):
        logger.debug(f"x_spec: {x_spec.shape},  y: {y.shape}")
        if i == 0:
            break
