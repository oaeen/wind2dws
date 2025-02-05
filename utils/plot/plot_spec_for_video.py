import os
import sys
from datetime import datetime
from pathlib import Path
import time
from datetime import datetime, timedelta
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from loguru import logger
from scipy.interpolate import RegularGridInterpolator as RGI
from concurrent.futures import ThreadPoolExecutor
from tqdm import trange


sys.path.append(os.getcwd())

from utils.metrics.spectra_info import generate_extend_dirs, generate_freq
from config import Config
from utils.plot.polar_spectrum import interpolate_and_extend_spectra, plot_spec


matplotlib.use("Agg")


def plot_spec_comparison(
    spec_predict,
    spec_true,
    filename,
    save_dir,
    title="12:00:00T Jan-01 2020",
    spec_freq_lim=0.4,
    vmax=2,
    colorbar_label=r"Spectral Density (m$^{2}$s)",
    contour=True,
    cb_ticks_interval=0.2,
    source_type=None,
    loc=None,
):
    """
    :param spec_predict: 预测的海浪谱，24个方向, 36个频率的海浪谱
    :param spec_true: 真实的海浪谱，24个方向, 36个频率的海浪谱
    :param filename: 保存图片的文件名
    :param save_dir: 保存图片的文件夹路径
    :return: 画图
    """

    def plot_single_spec(
        ax,
        spec,
        spec_freq_lim,
        vmax,
        colorbar_label,
        contour,
        cb_ticks_interval,
        source_type=source_type,
        title_label="",
    ):
        meshf, meshd, spec_extend = interpolate_and_extend_spectra(
            spec, source_type=source_type
        )

        plt.sca(ax)
        _ = plt.pcolormesh(
            meshd,
            meshf,
            spec_extend.transpose(),
            vmin=0,
            vmax=vmax,
            shading="gouraud",
            cmap="CMRmap_r",
        )
        cb = plt.colorbar(pad=0.10)
        cb.set_label(colorbar_label, fontsize=25)
        cb.ax.tick_params(labelsize=27)
        cb.locator = ticker.MultipleLocator(cb_ticks_interval)
        cb.update_ticks()

        if contour:
            contour_label = np.array(
                [0.01, 0.05, 0.1, 0.15, 0.3, 0.5, 0.9, 1.2, 1.5, 1.9]
            )
            _ = plt.contour(
                meshd,
                meshf,
                spec_extend.transpose(),
                contour_label,
                linewidths=1,
                colors="0.4",
            )

        ax.set_theta_direction(-1)
        ax.set_theta_zero_location("N")
        ax.set_ylim(0, spec_freq_lim)
        ax.set_rgrids(
            radii=np.linspace(0.1, spec_freq_lim, int(spec_freq_lim / 0.1)),
            labels=[f"{i:.1f}Hz" for i in np.arange(0.1, spec_freq_lim + 0.1, 0.1)],
            angle=158,
        )
        ax.xaxis.set_tick_params(pad=11)
        ax.set_title(title_label, ha="left", x=-0.39, y=0.85, fontsize=30)
        plt.grid(True)

    plt.rcdefaults()
    matplotlib.rc("xtick", labelsize=20)
    matplotlib.rc("ytick", color="0.25")
    matplotlib.rc("ytick", labelsize=20)
    matplotlib.rc("axes", lw=1.6)

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 14), subplot_kw={"polar": True})
    fig.suptitle(title, x=0.42, y=0.95, fontsize=30)

    plot_single_spec(
        ax=axes[0],
        spec=spec_true,
        spec_freq_lim=spec_freq_lim,
        vmax=vmax,
        colorbar_label=colorbar_label,
        contour=contour,
        cb_ticks_interval=cb_ticks_interval,
        source_type=source_type,
        title_label=f"{source_type}\n{loc}",
    )
    plot_single_spec(
        ax=axes[1],
        spec=spec_predict,
        spec_freq_lim=spec_freq_lim,
        vmax=vmax,
        colorbar_label=colorbar_label,
        contour=contour,
        cb_ticks_interval=cb_ticks_interval,
        source_type=source_type,
        title_label=f"{loc}\nPredicted",
    )

    Path.mkdir(Path(save_dir), parents=True, exist_ok=True)
    time_stamp = datetime.now().strftime("%m%d-%H%M%S")
    plt.savefig(f"{save_dir}/{filename}.png", dpi=150, bbox_inches="tight")
    plt.close("all")


def get_spec_time_list():
    """
    从提取的IOWAGA谱测试集数据中获取时间戳
    """
    # 测试集为 29年 * 0.2 = 5.8年的数据，取后5年的数据 ["2017", "2018", "2019", "2020", "2021"]
    # 后5年的数据为 5 * 365 * 8 + 8(闰一天) = 14608 个样本
    test_start_idx = 14608

    start_time = datetime(2017, 1, 1, 0, 0, 0)
    time_list = [start_time + timedelta(hours=3 * idx) for idx in range(test_start_idx)]
    return time_list


def get_specific_spec_samples(spec, target_year, config=Config()):

    logger.debug(f"绘制IOWAGA数据的指定年份样本谱图和平均谱图")
    time_list = get_spec_time_list()
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
            # 查找接近1月1日1时0分0秒的样本
            if (
                time.month == 1
                and time.day == 1
                and np.abs((time.hour + time.minute / 60)) <= 1
            ):
                logger.debug(
                    f"找到指定 {config.y_data_source}: {config.y_location}#{time} 的样本"
                )
                spec_samples = spec[idx : idx + 366 * 8]
                time_list = time_list[idx : idx + 366 * 8]

    spec_samples = np.array(spec_samples)
    time_list = np.array(time_list)
    logger.info(f"spec_samples.shape: {spec_samples.shape}")
    logger.info(f"time_list.shape: {time_list.shape}")

    return spec_samples, time_list


def process_plot(
    idx, t, spec_predict, spec_true, save_dir, vmax, cb_ticks_interval, source_type, loc
):
    title_format_time = t.strftime("%H:%M:%ST %b-%d %Y")
    filename_format_time = t.strftime("%m%d-%H%M%S")
    filename = f"{loc}_idx{idx:04d}_time_{filename_format_time}"
    print(f"Plotting {filename}")
    plot_spec_comparison(
        spec_predict[idx],
        spec_true[idx],
        filename,
        save_dir,
        title=title_format_time,
        vmax=vmax,
        cb_ticks_interval=cb_ticks_interval,
        source_type=source_type,
        loc=loc,
    )


if __name__ == "__main__":
    samples_plot_config = {
        "PointA": ("ERA5", 2, 0.4),
        "PointB": ("ERA5", 3, 0.5),
        "PointC": ("ERA5", 10, 2),
        "CDIP028": ("IOWAGA", 1.2, 0.2),
    }

    for loc in ["PointC"]:  # "PointA", "PointB", "PointC", "CDIP028"
        source_type, vmax, cb_ticks_interval = samples_plot_config[loc]
        logger.warning(
            f"{loc} spec sample cbar config: vmax = {vmax}, cb_ticks_interval = {cb_ticks_interval}"
        )

        isERA5 = ""
        inputRegion = "X"

        config = Config()
        config.y_location = loc
        comment = f"Wendy_Full_{loc}_run1"
        spec_predict = np.load(
            f"{os.getcwd()}/results/evaluate/{loc}/{comment}/y_predict.npy"
        )
        spec_true = np.load(
            f"{os.getcwd()}/results/evaluate/{loc}/{comment}/y_true.npy"
        )

        spec_predict, time_list = get_specific_spec_samples(spec_predict, 2020, config)
        spec_true, _ = get_specific_spec_samples(spec_true, 2020, config)

        replace_dict = {
            "PointA": "Point A",
            "PointB": "Point B",
            "PointC": "Point C",
            "CDIP028": "Point D",
        }

        save_dir = f"{os.getcwd()}/results/ForVideo/{replace_dict[loc]}/"
        Path.mkdir(Path(save_dir), parents=True, exist_ok=True)

        for idx in trange(len(time_list), desc="Processing plots"):
            t = time_list[idx]
            title_format_time = t.strftime("%H:%M:%ST %b-%d %Y")
            filename_format_time = t.strftime("%m%d-%H%M%S")
            filename = f"{replace_dict[loc]}_idx{idx:04d}_time_{filename_format_time}"
            # print(f"Plotting {filename}")
            plot_spec_comparison(
                spec_predict[idx],
                spec_true[idx],
                filename,
                save_dir,
                title=title_format_time,
                vmax=vmax,
                cb_ticks_interval=cb_ticks_interval,
                source_type=source_type,
                loc=replace_dict[loc],
            )
