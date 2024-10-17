import os
import pickle
import sys
from pathlib import Path

import numpy as np
from loguru import logger
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.getcwd())


def scale_spec2d(spec, spec_scale_bins="freq"):
    """
    对海浪谱进行 scale
    :param spec: 原始二维海浪谱数据
    :param spec_scale_bins: 缩放数据选择方式, "freq_and_dir" or "freq"
    :return: 缩放后的训练集数据, 缩放后的测试集数据, 保存的scaler字典
    """

    # 初始化保存scaler的字典
    scaler_dict = {}

    # 获取数据形状
    _, freq, direction = spec.shape

    spec_train = spec[: int(len(spec) * 0.8)]
    spec_test = spec[int(len(spec) * 0.8) :]

    logger.debug(f"spec_train.shape: {spec_train.shape}")
    logger.debug(f"spec_test.shape: {spec_test.shape}")

    if spec_scale_bins == None:
        logger.warning("not scale, return original data")
        return spec_train, spec_test, scaler_dict

    # 初始化缩放后的训练集和测试集数据
    spec_train_scaled = np.zeros_like(spec_train)
    spec_test_scaled = np.zeros_like(spec_test)

    for freq_idx in range(freq):
        scaler_obj = MinMaxScaler()

        if spec_scale_bins == "freq_and_dir":
            scaler_obj = scaler_obj.fit(spec_train[:, freq_idx, :])
            spec_train_scaled[:, freq_idx, :] = scaler_obj.transform(
                spec_train[:, freq_idx, :]
            )
            spec_test_scaled[:, freq_idx, :] = scaler_obj.transform(
                spec_test[:, freq_idx, :]
            )
            scaler_dict[freq_idx] = scaler_obj
        if spec_scale_bins == "freq":
            scaler_obj = scaler_obj.fit(
                spec_train[:, freq_idx, :].flatten().reshape(-1, 1)
            )
            spec_train_scaled[:, freq_idx, :] = scaler_obj.transform(
                spec_train[:, freq_idx, :].flatten().reshape(-1, 1)
            ).reshape(-1, direction)
            spec_test_scaled[:, freq_idx, :] = scaler_obj.transform(
                spec_test[:, freq_idx, :].flatten().reshape(-1, 1)
            ).reshape(-1, direction)
            scaler_dict[freq_idx] = scaler_obj

    return spec_train_scaled, spec_test_scaled, scaler_dict


def scale_spec2d_from_scaler(spec, scaler_dict, spec_scale_bins="freq"):
    """
    使用保存的scaler_dict将测试集数据进行缩放
    :param spec: 原始二维海浪谱数据
    :param spec_scale_bins: 缩放数据选择方式, "freq_and_dir" or "freq"
    :return: 缩放后的训练集数据, 缩放后的测试集数据, 保存的scaler字典
    """

    # 获取数据形状
    _, freq, direction = spec.shape

    if spec_scale_bins == None:
        logger.warning("not inverse, return original data")
        return spec

    # 初始化还原后的原始数据
    spec_scaled = np.zeros_like(spec)
    logger.info(f"scale_test_spec2d: {spec_scale_bins}")
    if spec_scale_bins == "freq_and_dir":
        for freq_idx in range(freq):
            scaler_obj = scaler_dict[freq_idx]
            spec_scaled[:, freq_idx, :] = scaler_obj.transform(spec[:, freq_idx, :])
    else:
        for freq_idx in range(freq):
            scaler_obj = scaler_dict[freq_idx]
            spec_scaled[:, freq_idx, :] = scaler_obj.transform(
                spec[:, freq_idx, :].flatten().reshape(-1, 1)
            ).reshape(-1, direction)
    return spec_scaled


def inverse_scale_spec2d(spec_scaled, scaler_dict, spec_scale_bins):
    """
    使用 scaler_dict 将缩放后的数据还原为原始数据
    :param spec_scaled: 缩放后的数据
    :param scaler_dict: 保存缩放器的字典
    :param spec_scale_bins: 缩放数据选择方式, "freq_and_dir" or "freq"
    :return: 还原后的原始数据
    """

    # 获取数据形状
    _, freq, direction = spec_scaled.shape

    if spec_scale_bins == None:
        logger.warning("not inverse, return original data")
        return spec_scaled

    # 初始化还原后的原始数据
    spec_unscaled = np.zeros_like(spec_scaled)
    # logger.info(f"inverse_scale_spec2d: {spec_scale_bins}")
    if spec_scale_bins == "minmax_freq_and_dir":
        for freq_idx in range(freq):
            scaler_obj = scaler_dict[freq_idx]
            spec_unscaled[:, freq_idx, :] = scaler_obj.inverse_transform(
                spec_scaled[:, freq_idx, :]
            )
    else:
        for freq_idx in range(freq):
            scaler_obj = scaler_dict[freq_idx]
            spec_unscaled[:, freq_idx, :] = scaler_obj.inverse_transform(
                spec_scaled[:, freq_idx, :].flatten().reshape(-1, 1)
            ).reshape(-1, direction)
    return spec_unscaled


def load_scaler(file_dir):
    scaler_path = f"{file_dir}/scaler.pkl"
    scaler = pickle.load(open(scaler_path, "rb"))
    logger.success(f"load scaler from {scaler_path}")
    return scaler


def log_spec2d(spec):
    """
    对海浪谱进行 scale
    :param spec: 原始二维海浪谱数据
    :param spec_scale_bins: 缩放数据选择方式, "freq_and_dir" or "freq"
    :return: 缩放后的训练集数据, 缩放后的测试集数据, 保存的scaler字典
    """

    # 初始化保存scaler的字典

    # 获取数据形状
    _, freq, direction = spec.shape

    spec = np.log(spec + 1)
    spec_train = spec[: int(len(spec) * 0.8)]
    spec_test = spec[int(len(spec) * 0.8) :]

    logger.debug(f"spec_train.shape: {spec_train.shape}")
    logger.debug(f"spec_test.shape: {spec_test.shape}")

    return spec_train, spec_test


def inverse_log_spec2d(spec):
    """
    使用 scaler_dict 将缩放后的数据还原为原始数据
    :return: 还原后的原始数据
    """

    # 获取数据形状
    _, freq, direction = spec.shape

    spec = np.exp(spec) - 1

    return spec
