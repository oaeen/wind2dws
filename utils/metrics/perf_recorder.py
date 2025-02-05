import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append(os.getcwd())

from config import Config


def save_loss_list(all_targets, all_outputs, config=Config()):
    mse_list = get_each_mse(all_targets, all_outputs)
    mae_list = get_each_mae(all_targets, all_outputs)
    correlation_list = get_each_cc(all_targets, all_outputs)
    r2_list = get_each_r2(all_targets, all_outputs)

    save_list_to_csv(mse_list, "mse", config)
    save_list_to_csv(mae_list, "mae", config)
    save_list_to_csv(correlation_list, "correlation", config)
    save_list_to_csv(r2_list, "r2", config)


def get_each_mae(all_targets, all_outputs):
    batch_size, freq, direction = all_targets.shape
    mae_list = np.zeros((freq, direction))
    for freq_idx in range(freq):
        for dir_idx in range(direction):
            all_targets_i = all_targets[:, freq_idx, dir_idx]
            all_outputs_i = all_outputs[:, freq_idx, dir_idx]
            mae = mean_absolute_error(all_targets_i, all_outputs_i)
            mae_list[freq_idx, dir_idx] = mae
    return mae_list


def get_each_mse(all_targets, all_outputs):
    batch_size, freq, direction = all_targets.shape
    mse_list = np.zeros((freq, direction))
    for freq_idx in range(freq):
        for dir_idx in range(direction):
            all_targets_i = all_targets[:, freq_idx, dir_idx]
            all_outputs_i = all_outputs[:, freq_idx, dir_idx]
            mse = mean_squared_error(all_targets_i, all_outputs_i)
            mse_list[freq_idx, dir_idx] = mse
    return mse_list


def get_each_cc(all_targets, all_outputs):
    batch_size, freq, direction = all_targets.shape
    correlation_list = np.zeros((freq, direction))
    # warnings.filterwarnings("error")

    for freq_idx in range(freq):
        for dir_idx in range(direction):
            all_targets_i = all_targets[:, freq_idx, dir_idx]
            all_outputs_i = all_outputs[:, freq_idx, dir_idx]
            # ConstantInputWarning
            correlation = 0
            try:
                correlation = np.corrcoef(all_outputs_i, all_targets_i)[1, 0]
            except:
                correlation = 0
                logger.warning("ConstantInputWarning")
                logger.warning(
                    f"freq{freq_idx:02d}dir{dir_idx:02d}avg_outputs_i: {np.mean(all_outputs_i)}"
                )
                logger.warning(
                    f"freq{freq_idx:02d}dir{dir_idx:02d}avg_targets_i: {np.mean(all_targets_i)}"
                )

            correlation_list[freq_idx, dir_idx] = correlation

    return correlation_list


def get_each_r2(all_targets, all_outputs):
    batch_size, freq, direction = all_targets.shape
    r2_list = np.zeros((freq, direction))
    for freq_idx in range(freq):
        for dir_idx in range(direction):
            all_targets_i = all_targets[:, freq_idx, dir_idx]
            all_outputs_i = all_outputs[:, freq_idx, dir_idx]
            r2 = r2_score(all_targets_i, all_outputs_i)
            r2_list[freq_idx, dir_idx] = r2
    return r2_list


def metrics(targets, outpusts, var_name):
    mse = mean_squared_error(targets, outpusts)
    mae = mean_absolute_error(targets, outpusts)
    R, _ = pearsonr(targets.flatten(), outpusts.flatten())
    logger.info(f"{var_name} MSE: {mse:.7f} | MAE: {mae:.7f} | R: {R:.4f}")
    return mse, mae, R


def report_perf(
    writer,
    current_epoch,
    run_time,
    train_loss,
    val_loss_dict,
    best_val_loss=None,
    best_val_loss_epoch=None,
    lr=None,
):
    """
    记录训练过程
    """
    val_loss = val_loss_dict["loss"]
    val_R = val_loss_dict["R"]
    logger.info(
        f"E{current_epoch} | LR: {lr} | Time: {run_time:.2f}s | Best in E{best_val_loss_epoch}: {best_val_loss*1e6:.0f}e-6"
    )
    logger.info(
        f"Train: {train_loss*1e6:.0f}e-6 | Val Loss: {val_loss*1e6:.0f}e-6 | R: {val_R:.4f}"
    )

    writer.add_scalar("Loss/train", train_loss, current_epoch)
    writer.add_scalar("Loss/val", val_loss, current_epoch)
    writer.add_scalar("R/val", val_R, current_epoch)


def backup_model_and_config(config=Config()):
    """
    将 network 模型 Python 文件 和 config 类下每个成员变量的名字和值保存到log文件夹下
    """
    log_dir = config.get_log_dir()
    model_file = f"{os.getcwd()}/models/{config.model_name}.py"
    shutil.copy(model_file, log_dir)

    config_dict = config.__dict__
    df = pd.DataFrame(config_dict.items(), columns=["name", "value"])
    df.to_csv(f"{log_dir}/config.csv", index=False)


def save_list_to_csv(data, var_name, config=Config()):
    """将各频率各方向的训练信息保存到 csv 文件中

    Args:
        data_list (_type_): 36 个频率， 总的信息+24个方向的信息
        model_name (_type_): 使用的模型名称
        comment (_type_): 用于区分不同的训练过程
    """
    log_dir = config.get_log_dir()
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv(f"{log_dir}/{var_name}.csv", index=False, header=False)
