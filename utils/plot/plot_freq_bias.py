from datetime import datetime
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

from utils.metrics.spectra_info import generate_freq

# 读取已经保存的频谱数据
locations = ["PointC"]  # "PointA", "PointB", "PointC", "CDIP028"

l_color = "#c93756"
g_color = "#f2be45"
x_color = "#21a675"


def get_freq_bias(loc, type):
    """
    loc: str, location name
    """

    bias_dir = f"Wendy{type}_Full_{loc}_run1"
    bias_filename = f"freq_bias.csv"
    bias_path = f"{os.getcwd()}/results/evaluate/{loc}/{bias_dir}/{bias_filename}"
    print(bias_path)
    bias = np.genfromtxt(bias_path, delimiter=",")

    return bias


for loc in locations:
    freq_large = get_freq_bias(loc, "_Large")
    freq_local = get_freq_bias(loc, "_Local")
    freq_x = get_freq_bias(loc, "")

    freq_indices = np.arange(1, len(freq_large) + 1)

    freq = None
    if loc in ["PointA", "PointB", "PointC"]:
        freq = generate_freq(spec_source_type="ERA5")

    if loc == "CDIP028":
        freq = generate_freq(spec_source_type="IOWAGA")

    bar_width = 0.25

    # clear
    plt.clf()

    plt.figure(figsize=(12, 6))
    plt.bar(
        freq_indices - bar_width,
        freq_local,
        width=bar_width,
        label="Local Scale",
        color=l_color,
    )
    plt.bar(
        freq_indices, freq_large, width=bar_width, label="Large Scale", color=g_color
    )
    plt.bar(
        freq_indices + bar_width,
        freq_x,
        width=bar_width,
        label="Double Scales",
        color=x_color,
    )

    plt.xlabel("Frequency (Hz)", fontdict={"fontsize": 18})
    plt.ylabel("Bias", fontdict={"fontsize": 18})

    replace_dict = {
        "PointA": "Point A",
        "PointB": "Point B",
        "PointC": "Point C",
        "CDIP028": "Point D",
    }
    title = f"Different Frequency Bias at {replace_dict[loc]}"

    plt.title(title, fontdict={"fontsize": 18})
    plt.xticks(freq_indices, labels=np.around(freq, 3), rotation=45, fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(prop={"size": 16}, framealpha=0.3)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    all_data = np.concatenate([freq_local, freq_large, freq_x])
    ymin, ymax = np.min(all_data), np.max(all_data)
    print(f"ymin: {ymin}, ymax: {ymax}")
    y_lim_config = {
        "PointA": (-0.4, 0.01),
        "PointB": (-0.1, 0.001),
        "PointC": (-0.25, 0.02),
        "CDIP028": (-0.05, 0.001),
    }
    plt.ylim(y_lim_config[loc])
    plt.gca().invert_yaxis()
    plt.tight_layout()
    # plt.show()

    # 保存图片
    save_dir = f"{os.getcwd()}/results/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    time_stamp = datetime.now().strftime("%m%d-%H%M%S")
    plt.savefig(
        f"{save_dir}freq_bias_{replace_dict[loc]}_l{l_color}_g{g_color}_x{x_color}_{time_stamp}.png",
        dpi=600,
    )
