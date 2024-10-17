import os
import sys
import time
from datetime import datetime

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import pearsonr

sys.path.append(os.getcwd())

from config import Config
from utils.metrics.perf_recorder import *
from utils.metrics.spec_metrics import Spec_Metrics
from utils.metrics.spectra_info import *
from utils.plot.polar_spectrum import plot_spec

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def load_data(
    data_path,
    X_location,
    y_location,
):
    """
    按照batch_size & window_size划分数据集, 并加载到iterableDataset中
    返回train_dataloader, test_dataloader
    """

    X_train = np.load(f"{data_path}/{X_location}/train.npy")
    X_test = np.load(f"{data_path}/{X_location}/test.npy")

    y_train = np.load(f"{data_path}/{y_location}/train.npy")
    y_test = np.load(f"{data_path}/{y_location}/test.npy")

    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    return X, y


def load_loss_csv(loss_name, config=Config()):
    loss_data_path = f"{config.get_log_dir()}/{loss_name}.csv"
    loss = np.loadtxt(loss_data_path, delimiter=",")
    loss = loss[:, :24]
    return loss


def calculate_freq_cc(targets, outputs):
    _sample, freq_num, direction_num = targets.shape
    freq_cc = []
    for i in range(freq_num):
        X_i = targets[:, i, :]
        y_i = outputs[:, i, :]
        correlation_coefficient, _ = pearsonr(X_i.flatten(), y_i.flatten())
        freq_cc.append(correlation_coefficient)

    freq_cc = np.array(freq_cc)
    return freq_cc


def plot_freq_cc(freq_cc, save_dir, filename, color="#FA7F6F", label="", start_freq=1):
    freq_num = len(freq_cc)

    frequencies = np.arange(start_freq, freq_num + 1)

    plt.bar(frequencies, freq_cc, color=color, label=label, alpha=0.8)

    plt.title(f"{filename} freq correlation coefficients")
    plt.xlabel("Frequency")
    plt.ylabel("Correlation coefficient")
    plt.legend()
    time = datetime.now().strftime("%m%d-%H%M%S")
    plt.savefig(f"{save_dir}/{filename}_{time}.png")
    # plt.show()
    plt.close("all")


def plot_freq_cc_compare(
    X_location, y_location, true_freq_cc, comp_freq_cc, label1="real", label2=""
):
    frequencies = np.arange(1, 37)

    plt.bar(frequencies, comp_freq_cc, color="#FA7F6F", label=label2, alpha=0.6)
    plt.bar(frequencies, true_freq_cc, color="#82B0D2", label=label1, alpha=0.6)

    plt.title(f"{X_location} -> {y_location} freq correlation coefficients")
    plt.xlabel("Frequency")
    plt.ylabel("Correlation coefficient")
    plt.legend()
    plt.savefig(f"{os.getcwd()}/results/{y_location}/pictures/freq_cc.png")
    plt.show()
    plt.close("all")


def plot_spec_loss(save_dir):
    """
    绘制两点间数据的频谱 loss
    """
    X_location = "46219"
    # X_location = "W1200N315"
    y_locations = ["CDIP028", "CDIP045", "CDIP093", "CDIP107"]
    for y_location in y_locations:
        X, y = load_data(X_location=X_location, y_location=y_location)

        correlation_coefficient, _ = pearsonr(X.flatten(), y.flatten())
        logger.info(f"{y_location} correlation_coefficient: {correlation_coefficient}")

        spec_corr = np.zeros((36, 24))
        for freq_idx in range(36):
            freq_cc = []
            for dir_idx in range(24):
                X_i = X[:, freq_idx, dir_idx]
                y_i = y[:, freq_idx, dir_idx]
                correlation_coefficient, _ = pearsonr(X_i.flatten(), y_i.flatten())
                freq_cc.append(correlation_coefficient)
            spec_corr[freq_idx, :] = freq_cc

        plot_spec(
            spec_corr,
            save_dir=save_dir,
            filename=f"{X_location} → {y_location}",
            colorbar_label=f"MinMax {X_location} correlation coefficient",
            contour=False,
            vmax=1,
            source_type=config.y_data_source,
        )


def plot_freq_hist(y_predict_unscale, y_true_unscale, config=Config()):
    _sample, freq_num, direction_num = y_predict_unscale.shape
    spec_desc = get_spec_desc()

    for freq_idx in range(freq_num):
        y_predict_unscale_freq = y_predict_unscale[:, freq_idx, :].flatten()
        y_true_unscale_freq = y_true_unscale[:, freq_idx, :].flatten()
        spec_desc["data_type"] = f"spec_freq{freq_idx + 1}"
        plot_hist2d(
            y_true_unscale_freq,
            y_predict_unscale_freq,
            spec_desc,
            config=config,
        )


def plot_direction_hist(y_predict_unscale, y_true_unscale, config=Config()):
    _sample, freq_num, direction_num = y_predict_unscale.shape
    spec_desc = get_spec_desc()

    for dir_idx in range(direction_num):
        y_predict_unscale_dir = y_predict_unscale[:, :, dir_idx].flatten()
        y_true_unscale_dir = y_true_unscale[:, :, dir_idx].flatten()
        spec_desc["data_type"] = f"spec_dir{dir_idx + 1}"
        plot_hist2d(
            y_true_unscale_dir,
            y_predict_unscale_dir,
            spec_desc,
            config=config,
        )


def plot_metrics(y_predict_unscale, y_true_unscale, config=Config()):
    _sample, freq, direction = y_predict_unscale.shape
    metrics = Spec_Metrics(
        freq_num=freq, direction_num=direction, source_type=config.y_data_source
    )

    (
        swh_pre,
        mwd_pre,
        mwp_minus1_pre,
        mwp1_pre,
        mwp2_pre,
    ) = metrics.integral_predict_spec_parameters(y_predict_unscale)
    (
        swh_true,
        mwd_true,
        mwp_minus1_true,
        mwp1_true,
        mwp2_true,
    ) = metrics.integral_predict_spec_parameters(y_true_unscale)

    # 角度矫正
    mwd_pre[np.where((mwd_true - mwd_pre) > 180)[0]] = (
        mwd_pre[np.where((mwd_true - mwd_pre) > 180)[0]] + 360
    )
    mwd_true[np.where((mwd_true - mwd_pre) < -180)[0]] = (
        mwd_true[np.where((mwd_true - mwd_pre) < -180)[0]] + 360
    )
    # 绘制直方图
    plot_hist2d(y_true_unscale, y_predict_unscale, get_spec_desc(), config=config)
    plot_hist2d(swh_true, swh_pre, get_swh_desc(), config=config)
    plot_hist2d(mwd_true, mwd_pre, get_mwd_desc(), config=config)
    plot_hist2d(mwp_minus1_true, mwp_minus1_pre, get_mwp_minus1_desc(), config=config)
    plot_hist2d(mwp1_true, mwp1_pre, get_mwp1_desc(), config=config)
    plot_hist2d(mwp2_true, mwp2_pre, get_mwp2_desc(), config=config)


def plot_hist2d(y_true, y_predict, data_description, config=Config()):
    y_true = y_true.flatten()
    y_predict = y_predict.flatten()

    # 将 NaN 值替换为 0
    y_true = np.nan_to_num(y_true)
    y_predict = np.nan_to_num(y_predict)

    data_type = data_description["data_type"]
    max_value = data_description["max_value"]
    vmax = data_description["vmax"]
    xlabel_text = data_description["xlabel_text"]
    ylabel_text = data_description["ylabel_text"]
    unit_text = data_description["unit_text"]

    if data_type == "swh" and config.y_location == "CDIP067":
        """开阔海域的浪高范围较大, 为了更好的绘图效果, 修改最大值为 8"""
        max_value = 8

    metrics = Spec_Metrics(source_type=config.y_data_source)
    rmse, bias, corrcoef = metrics.evaluate_predict_spec_loss(
        y_true, y_predict, data_type
    )
    bias = f"{bias:.2g}" if np.abs(bias) < 1 else f"{bias:.2f}"
    upper_left_text = (
        f"RMSE = {rmse:.2f} {unit_text}\nBias = {bias} {unit_text}\nR = {corrcoef:.2f}"
    )

    plt.rcdefaults()
    plt.clf()
    plt.figure(figsize=(6, 6))

    fig, ax = plt.subplots()

    plt.hist2d(
        y_true,
        y_predict,
        bins=120,
        cmap=plt.cm.viridis,
        norm=mcolors.LogNorm(vmax=vmax),
        range=[[0, max_value], [0, max_value]],
    )
    plt.plot([0, max_value], [0, max_value])
    plt.axis("equal")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True)
    cb = plt.colorbar()
    cb.set_label(r"Data density", fontsize=13)
    cb.ax.tick_params(labelsize=14)  # Increase colorbar ticks font size
    cb.update_ticks()

    ax.tick_params(axis="both", which="major", labelsize=14)

    plt.text(0.05, 0.80, upper_left_text, transform=ax.transAxes, fontsize=14)
    plt.xlabel(xlabel_text, fontsize=13)
    plt.ylabel(ylabel_text, fontsize=13)
    evaluate_save_dir = config.get_evaluate_figure_save_dir()
    filename = f"{evaluate_save_dir}/hist2d_{data_type}_{datetime.now().strftime('%m%d-%H%M%S')}.png"
    plt.savefig(filename, dpi=600, bbox_inches="tight")
    plt.close("all")
    logger.success(f"savefig: {filename}")


def plot_line_comparison(pre, real, data_description, config=Config()):
    data_type = data_description["data_type"]
    xlabel_text = data_description["xlabel_text"]
    ylabel_text = data_description["ylabel_text"]
    plt.rcdefaults()
    # 创建一个新的图形
    plt.figure(figsize=(10, 6))

    # 绘制折线图
    plt.plot(real, label="True", linestyle="-", linewidth=1.2, color="gray", alpha=1)
    plt.plot(
        pre, label="Predict", linestyle="--", linewidth=1, color="#ff7f0e", alpha=0.95
    )

    # 添加图例
    plt.legend()

    # 添加横坐标和纵坐标的标签文本
    plt.xlabel(xlabel_text, fontsize=14)
    plt.ylabel(ylabel_text, fontsize=14)

    # 设置横坐标的刻度和标签
    date_range = pd.date_range(start="2017-01-01", end="2017-12-31", freq="1M")
    ax = plt.gca()
    ax.xaxis.set_major_locator(
        plt.FixedLocator(np.linspace(0, len(pre), len(date_range)))
    )
    ax.xaxis.set_major_formatter(plt.FixedFormatter(date_range.strftime("%Y.%m")))
    plt.xticks(rotation=60)

    # 显示图形
    plt.grid(True)
    plt.title("predict vs real comparison")

    evaluate_save_dir = config.get_evaluate_figure_save_dir()
    filename = f"{evaluate_save_dir}/comparison_{data_type}_{datetime.now().strftime('%m%d-%H%M%S')}.png"
    plt.savefig(filename, dpi=600, bbox_inches="tight")
    logger.success(f"savefig: {filename}")
    plt.close("all")


if __name__ == "__main__":
    X_location, y_location = "46219", "CDIP028"

    config = Config()
    network_type = config.network_type
    comment = config.comment

    loss_dir = f"{os.getcwd()}/{config.y_location}/{comment}/loss/"

    mse_loss = load_loss_csv(loss_name="mse", config=config)
    mae_loss = load_loss_csv(loss_name="mae", config=config)
    cc_loss = load_loss_csv(loss_name="correlation", config=config)
    r2_loss = load_loss_csv(loss_name="r2", config=config)

    # 扩展为 36 * 24
    mse_loss = np.concatenate((mse_loss, np.zeros((6, 24))), axis=0)
    mae_loss = np.concatenate((mae_loss, np.zeros((6, 24))), axis=0)
    cc_loss = np.concatenate((cc_loss, np.zeros((6, 24))), axis=0)
    r2_loss = np.concatenate((r2_loss, np.zeros((6, 24))), axis=0)

    plot_spec(
        mse_loss,
        save_dir=loss_dir,
        filename=f"{comment}_mse",
        colorbar_label="Mean Square Error",
        vmax=np.max(mse_loss),
        source_type=config.y_data_source,
    )
    plot_spec(
        mae_loss,
        save_dir=loss_dir,
        filename=f"{comment}_mae",
        colorbar_label="Mean Absolute Error",
        vmax=np.max(mae_loss),
        source_type=config.y_data_source,
    )
    plot_spec(
        cc_loss,
        save_dir=loss_dir,
        filename=f"{comment}_cc",
        colorbar_label="Correlation Coefficient",
        vmax=1,
        source_type=config.y_data_source,
    )

    plot_spec(
        r2_loss,
        save_dir=loss_dir,
        filename=f"{comment}_r2",
        colorbar_label="R2 Score",
        vmax=np.max(r2_loss),
        source_type=config.y_data_source,
    )
