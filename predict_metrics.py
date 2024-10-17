import os
import sys
import time

import numpy as np
import torch
from loguru import logger

sys.path.append(os.getcwd())

from config import Config
from models.model import load_model
from utils.data_loaders.helper import get_dataloader
from utils.metrics.perf_recorder import *
from utils.metrics.spectra_info import correct_spec_direction
from utils.plot.polar_spectrum import plot_spec, plot_spec_list, plot_spec_list_2d
from utils.preprocess.extract.spec_fetcher import *
from utils.preprocess.prepare.scale_spec import inverse_scale_spec2d, load_scaler


@torch.no_grad()
def predict(model, dataloader, config=Config()):
    """
    使用训练好的模型预测一个点的36个频率24个方向的海浪谱, 并进行逆变换
    """
    model.eval()

    all_predict = []
    all_targets = []

    for pw, cw, targets in dataloader:
        pw = pw.to(config.device)
        cw = cw.to(config.device)
        targets = targets.to(config.device)
        outputs = model(pw, cw)

        all_predict.append(outputs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    all_predict = np.concatenate(all_predict, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    return all_predict, all_targets


def process_predict_spec2d(config=Config()):
    predict_start_time = time.time()
    model, best_epoch, best_loss = load_model(config)

    _train_dataloader, test_dataloader = get_dataloader(config)

    y_predict, y_true = predict(model, test_dataloader, config=config)

    predict_end_time = time.time()

    logger.success(f"predict running time: {predict_end_time - predict_start_time}")

    # 还原缩放后的数据
    y_scaler = load_scaler(config.get_y_data_dir())
    y_predict = inverse_scale_spec2d(y_predict, y_scaler, config.y_scale_desc)
    y_true = inverse_scale_spec2d(y_true, y_scaler, config.y_scale_desc)

    y_predict = correct_spec_direction(y_predict, config.y_data_source)
    y_true = correct_spec_direction(y_true, config.y_data_source)

    ############################### 检验 #####################################

    # 计数小于零的个数
    logger.warning(f"y_predict < 0 num: {(y_predict < 0).sum()}")
    logger.warning(f"y_predict < 0 avg: {np.mean(y_predict[y_predict < 0])}")
    logger.warning(f"y_true < 0 num: {(y_true < 0).sum()}")

    logger.debug(f"y_predict.shape: {y_predict.shape}")
    logger.debug(f"y_true.shape: {y_true.shape}")

    return y_predict, y_true


def plot_output_spec_samples(y_predict, y_true, config=Config()):
    # specific for CDIP028, CDIP045, CDIP067;
    # CDIP浮标数据中中 093, 107 浮标点时间不够长, 无法取到 2017 年以后的数据
    target_year = 2020

    samples_save_dir = config.get_samples_figure_save_dir()

    month_indices = [1, 4, 7, 10]
    y_predict_avg, y_predict_samples = get_specific_spec_samples(
        y_predict, target_year, month_indices, config
    )
    y_true_avg, y_true_samples = get_specific_spec_samples(
        y_true, target_year, month_indices, config
    )

    vmax = 1.2
    cb_ticks_interval = 0.2

    if config.y_location == "PointA":
        vmax = 2
        cb_ticks_interval = 0.4
        logger.warning(f"{config.y_location} spec vmax changing to: {vmax}")

    if config.y_location == "PointB":
        vmax = 3
        cb_ticks_interval = 0.5
        logger.warning(f"{config.y_location} spec vmax changing to: {vmax}")

    plot_spec(
        y_predict_avg,
        f"predict_avg",
        samples_save_dir,
        source_type=config.y_data_source,
        vmax=vmax,
        cb_ticks_interval=cb_ticks_interval,
    )
    plot_spec(
        y_true_avg,
        f"true_avg",
        samples_save_dir,
        source_type=config.y_data_source,
        vmax=vmax,
        cb_ticks_interval=cb_ticks_interval,
    )

    logger.debug(f"y_predict_samples.shape: {y_predict_samples.shape}")
    logger.debug(f"y_true_samples.shape: {y_true_samples.shape}")

    filename_list = [f"month{idx:02d}_day01_year{target_year}" for idx in month_indices]

    plot_spec_list(
        y_predict_samples,
        filename_list,
        samples_save_dir,
        filename_appendix=f"_predict",
        source_type=config.y_data_source,
        vmax=vmax,
        cb_ticks_interval=cb_ticks_interval,
    )
    plot_spec_list(
        y_true_samples,
        filename_list,
        samples_save_dir,
        filename_appendix=f"_true",
        source_type=config.y_data_source,
        vmax=vmax,
        cb_ticks_interval=cb_ticks_interval,
    )


if __name__ == "__main__":
    config = Config()
