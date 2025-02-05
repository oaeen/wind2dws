import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from loguru import logger
from scipy.interpolate import RegularGridInterpolator as RGI

sys.path.append(os.getcwd())

from utils.metrics.spectra_info import generate_extend_dirs, generate_freq
from config import Config


def interpolate_and_extend_spectra(
    spec,
    freq_new_num=72,
    dir_new_num=72,
    source_type=None,
):
    """
    对海浪谱进行插值处理, cubic插值后的海浪谱  (36, 24) -> (64, 72)
    并在方向上拓展，解决绘制极坐标图像时, 0°=360° 可能导致的不连续问题
    """
    spec = np.nan_to_num(spec)
    freq_n, dir_n = spec.shape
    spec = np.concatenate((spec, spec[:, 0:1]), axis=1)
    freq_ori = generate_freq(spec_source_type=source_type)
    dirs_ori = generate_extend_dirs(n=dir_n + 1, spec_source_type=source_type)

    interpolator = RGI((dirs_ori, freq_ori), spec.T, method="cubic", bounds_error=False)

    freq_new = generate_freq(new_n=freq_new_num, spec_source_type=source_type)
    dirs_new = generate_extend_dirs(dir_new_num + 1, spec_source_type=source_type)
    meshf, meshd = np.meshgrid(freq_new, dirs_new)
    spec = interpolator((meshd, meshf)).T

    return meshf, meshd, spec


def plot_spec1d(spec_pred2d, spec_true2d, filename, save_dir, source_type=None):
    freq = generate_freq(spec_source_type=source_type)
    # 生成方向数组（假设均匀分布）
    n_dir = spec_pred2d.shape[1]
    theta = np.linspace(0, 2 * np.pi, n_dir, endpoint=False)

    # 数值积分计算一维谱（使用梯形积分法）
    pred_1d = np.trapz(spec_pred2d, x=theta, axis=1)
    true_1d = np.trapz(spec_true2d, x=theta, axis=1)

    # 创建画布
    plt.figure(figsize=(8, 8))

    # 绘制谱线
    plt.semilogy(freq, true_1d, label="True", color="navy", linewidth=2, alpha=0.8)

    plt.semilogy(
        freq,
        pred_1d,
        label="Predicted",
        color="crimson",
        linewidth=2,
        linestyle="--",
        alpha=0.8,
    )

    # 图形修饰
    # plt.title("1D Wave Spectrum Comparison", fontsize=16, pad=20)
    plt.xlabel("Frequency (Hz)", fontsize=30)
    plt.ylabel("Energy Density (m²s)", fontsize=30)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    # legend 固定在右上，50%透明度
    # plt.legend(fontsize=30, loc="upper right", framealpha=0.5)
    # legend 固定在右下，50%透明度
    plt.legend(fontsize=30, loc="lower right", framealpha=0.5)
    plt.xlim([0, freq.max()])

    # 自动调整布局
    plt.tight_layout()

    # 创建保存路径
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_spec(
    spec,
    filename,
    save_dir,
    spec_freq_lim=0.4,
    vmax=2,
    colorbar_label=r"Spectral Density (m$^{2}$s)",
    contour=True,
    source_type=None,
    cb_ticks_interval=0.2,
):
    """
    :param title: 图片标题
    :param wave_spectra: 24个方向, 36个频率的海浪谱
    :return: 画图
    """
    meshf, meshd, spec_extend = interpolate_and_extend_spectra(
        spec, source_type=source_type
    )

    plt.rcdefaults()
    # matplotlib.rc('font', family='fantasy')
    matplotlib.rc("xtick", labelsize=20)
    matplotlib.rc("ytick", color="0.25")
    matplotlib.rc("ytick", labelsize=20)
    matplotlib.rc("axes", lw=1.6)

    _ = plt.figure(None, (8.5, 7))
    ax1 = plt.subplot(111, polar=True)
    plt.rc("grid", color="0.3", linewidth=0.8, linestyle="dotted")
    # 绘制极坐标图
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
    cb.set_label(colorbar_label, fontsize=22)
    cb.ax.tick_params(labelsize=24)
    cb.locator = ticker.MultipleLocator(cb_ticks_interval)
    cb.update_ticks()

    if contour:
        contour_label = np.array([0.01, 0.05, 0.1, 0.15, 0.3, 0.5, 0.9, 1.2, 1.5, 1.9])
        # 绘制等高线
        _ = plt.contour(
            meshd,
            meshf,
            spec_extend.transpose(),
            contour_label,
            linewidths=1,
            colors="0.4",
        )

    ax1.set_theta_direction(-1)
    ax1.set_theta_zero_location("N")
    ax1.set_ylim(0, spec_freq_lim)
    ax1.set_rgrids(
        radii=np.linspace(0.1, spec_freq_lim, int(spec_freq_lim / 0.1)),
        labels=[f"{i:.1f}Hz" for i in np.arange(0.1, spec_freq_lim + 0.1, 0.1)],
        angle=158,
    )
    ax1.xaxis.set_tick_params(pad=11)

    plt.grid(True)

    Path.mkdir(Path(save_dir), parents=True, exist_ok=True)
    time_stamp = datetime.now().strftime("%m%d-%H%M%S")

    plt.savefig(f"{save_dir}/{filename}_{time_stamp}.png", dpi=600, bbox_inches="tight")
    plt.close("all")


def plot_spec_list(
    spec_list,
    filename_list,
    save_dir,
    filename_appendix="",
    source_type=None,
    vmax=1.2,
    cb_ticks_interval=0.2,
):
    for spec, filename in zip(spec_list, filename_list):
        plot_spec(
            spec,
            filename=filename + filename_appendix,
            save_dir=save_dir,
            source_type=source_type,
            vmax=vmax,
            cb_ticks_interval=cb_ticks_interval,
        )


def plot_spec_list_2d(
    spec_list,
    filename_list,
    save_dir,
    filename_appendix="",
):
    for spec, filename in zip(spec_list, filename_list):
        plot_spec_2d(
            spec,
            filename=filename + filename_appendix,
            save_dir=save_dir,
        )


def plot_spec_2d(
    spec,
    filename,
    save_dir,
):
    """
    绘制二维谱图, 并保存, cmap="CMRmap_r"
    """
    plt.rcdefaults()
    spec = np.swapaxes(spec, 0, 1)

    plt.axis("off")
    plt.imshow(spec, cmap="CMRmap_r", vmin=0, vmax=2)
    plt.savefig(f"{save_dir}/{filename}.png", dpi=600, bbox_inches="tight")
    plt.close("all")


if __name__ == "__main__":
    config = Config()
    location = "PointB"
    config.y_location = location
    config.y_data_source = "ERA5"

    test_file = f"{config.get_y_data_dir()}/test.npy"

    wave_spectra = np.load(test_file, allow_pickle=True)
    save_dir = f"{os.getcwd()}/results/test/{location}"
    time = 0
    for time in range(0, 600, 23):
        plot_spec(
            wave_spectra[time],
            filename=f"time{time}",
            save_dir=save_dir,
            source_type=config.y_data_source,
        )
