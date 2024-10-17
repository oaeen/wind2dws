from datetime import datetime
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

from utils.metrics.spectra_info import generate_freq

# 读取已经保存的频谱数据
locations = ["PointA", "PointB", "CDIP028"]

l_color = "#c93756"
g_color = "#f2be45"
x_color = "#21a675"


def get_freq_cc(loc, type):
    """
    loc: str, location name
    type: str, "G", global  or "L" local or "X" mixed
    """
    # assert type in ["G", "L", "X"]

    cc_dir = f"WindNet_{type}1010_Seele_{loc}_scale_minmax_freq_and_dir_TP80"
    cc_filename = f"freq_cc.csv"
    cc_path = f"{os.getcwd()}/results/evaluate/{loc}/{cc_dir}/{cc_filename}"
    print(cc_path)
    cc = np.genfromtxt(cc_path, delimiter=",")

    return cc


for loc in locations:
    type_prefix = "ERA5_" if loc != "CDIP028" else ""
    freq_g = get_freq_cc(loc, type_prefix + "G")
    freq_l = get_freq_cc(loc, type_prefix + "L")
    freq_x = get_freq_cc(loc, type_prefix + "X")

    freq_indices = np.arange(1, len(freq_g) + 1)

    freq = None
    if loc in ["PointA", "PointB"]:
        freq = generate_freq(spec_source_type="ERA5")

    if loc == "CDIP028":
        freq = generate_freq(spec_source_type="IOWAGA")

    bar_width = 0.25

    # clear
    plt.clf()

    plt.figure(figsize=(12, 6))
    plt.bar(
        freq_indices - bar_width,
        freq_l,
        width=bar_width,
        label="Local Scale",
        color=l_color,
    )
    plt.bar(freq_indices, freq_g, width=bar_width, label="Large Scale", color=g_color)
    plt.bar(
        freq_indices + bar_width,
        freq_x,
        width=bar_width,
        label="Double Scales",
        color=x_color,
    )

    plt.xlabel("Frequency (Hz)", fontdict={"fontsize": 14})
    plt.ylabel("Correlation Coefficient", fontdict={"fontsize": 14})

    replace_dict = {
        "PointA": "Point A",
        "PointB": "Point B",
        "CDIP028": "Point C",
    }
    title = f"Different Frequency Correlation Coefficients at {replace_dict[loc]}"

    plt.title(title, fontdict={"fontsize": 14})
    plt.xticks(freq_indices, labels=np.around(freq, 3), rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(prop={"size": 12}, framealpha=0.3)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    # plt.show()

    # 保存图片
    save_dir = f"{os.getcwd()}/results/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    time_stamp = datetime.now().strftime("%m%d-%H%M%S")
    plt.savefig(
        f"{save_dir}freq_cc_{replace_dict[loc]}_l{l_color}_g{g_color}_x{x_color}_{time_stamp}.png",
        dpi=600,
    )
