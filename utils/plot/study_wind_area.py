from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os


class StudyArea:
    def __init__(self) -> None:
        # 添加实心圆点和注释
        self.Point_locations = {
            "Point A": (115, 15),
            "Point B": (240, 5),
            "Point C": (-118.6 + 360, 33.9),
        }


point_color = "#e83929"
rect_color = "#16a951"


def plot_study_area(lon_start, lon_end, lat_start, lat_end, study_area=StudyArea()):
    fig, ax = plt.subplots(
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=180)}
    )
    ax.set_extent([lon_start, lon_end, lat_start, lat_end])

    ax.set_ylim(lat_start, lat_end)

    ax.add_feature(cfeature.LAND, color="lightgrey")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
    ax.add_feature(cfeature.OCEAN, color="#f0f0f4")

    lat_formatter = FuncFormatter(format_lat_tick)
    lon_formatter = FuncFormatter(format_lon_tick)

    ax.set_xticks(
        np.arange(int(lon_start), int(lon_end + 1), 30), crs=ccrs.PlateCarree()
    )
    ax.set_yticks(np.arange(-70, 71, 10), crs=ccrs.PlateCarree())

    ax.set_aspect(1.0)

    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude", fontsize=8)
    ax.tick_params(axis="both", labelsize=8, direction="in")
    ax.set_title(
        "Target Points and Wind Field Regions",
        fontdict={"fontsize": 13, "fontweight": "bold", "fontname": "Calibri"},
        loc="center",
    )

    for place, (lon, lat) in study_area.Point_locations.items():
        ax.scatter(
            lon,
            lat,
            color=point_color,
            s=18,
            zorder=10,
            edgecolors="black",
            linewidths=0.5,
            transform=ccrs.PlateCarree(),
        )  # 添加实心圆点
        ax.annotate(
            place,
            (lon, lat),
            textcoords="offset points",
            xytext=(-19, -4),
            ha="center",
            transform=ccrs.PlateCarree(),
            fontsize=9,
        )

    # 绘制绿色矩形
    for point in study_area.Point_locations.values():
        rect_coords = [
            (point[0] - 20, point[1] + 20),
            (point[0] + 20, point[1] + 20),
            (point[0] + 20, point[1] - 20),
            (point[0] - 20, point[1] - 20),
            (point[0] - 20, point[1] + 20),
        ]
        rect_lons, rect_lats = zip(*rect_coords)
        ax.plot(
            rect_lons,
            rect_lats,
            color=rect_color,
            transform=ccrs.PlateCarree(),
            linewidth=1.0,
        )

    # 为图例创建虚拟的点和矩形
    point_legend = ax.scatter(
        [],
        [],
        color=point_color,
        edgecolors="black",
        linewidths=0.5,
        label="Target Point",
    )
    # rect_legend = ax.plot(
    #     [], [], color=rect_color, linewidth=1.2, label="Local Wind Field Region"
    # )

    rect_legend_patch = mpatches.Rectangle(
        (0, 0),
        1,
        1,
        fill=False,
        edgecolor=rect_color,
        linewidth=1.0,
        label="Local Wind Field Region",
    )

    # 创建正方形的矩形，没有填充
    square_legend_patch = mpatches.Rectangle(
        (0, 0),
        1,
        1,
        fill=False,
        edgecolor=rect_color,
        linewidth=1.0,
        label="Local Wind Field Region",
    )

    ax.legend(
        handles=[point_legend, square_legend_patch],
        loc="lower right",
        fontsize=8,
        frameon=True,
        framealpha=0.1,
        edgecolor="black",
        bbox_to_anchor=(1, 0.02),
        handler_map={mpatches.Rectangle: HandlerPatch(patch_func=make_legend_square)},
        markerscale=0.8,  # 缩小图例符号的大小
        handletextpad=0.15,  # 图例符号和文本之间的距离
    )
    time_stamp = datetime.now().strftime("%m%d-%H%M%S")
    plt.savefig(
        f"{os.getcwd()}/results/study_wind_{time_stamp}.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close("all")


def format_lat_tick(value, pos):

    n_s = ""
    if abs(value - 90) > 1e-6:
        if value > 0:
            n_s = "N"
        if value < 0:
            value = -value
            n_s = "S"

    value = int(value + 0.5)
    # value = f"{value:.0f}" if value.is_integer() else f"{value:.1f}"
    return f"{value}°{n_s}"


def format_lon_tick(value, pos):
    print(f"value: {value}")
    value = value + 180

    e_w = ""
    if abs(value - 180) > 1e-6:
        if value < 180:
            e_w = "E"
        if value > 180:
            value = 360 - value
            e_w = "W"

    value = int(value + 0.5)
    # value = f"{value:.0f}" if value.is_integer() else f"{value:.1f}"
    return f"{value}°{e_w}"


def make_legend_square(
    legend, orig_handle, xdescent, ydescent, width, height, fontsize
):
    # 使用 scale_factor 来控制正方形的大小
    scale_factor = 0.32  # 缩小到原来的50%

    # 调整 xdescent 和 ydescent 以控制正方形的位置，使其居中显示
    xdescent_adjusted = xdescent + (width - width * scale_factor) / 2 + 0.5
    ydescent_adjusted = ydescent + (height - width * scale_factor) / 2

    # 创建正方形
    square = mpatches.Rectangle(
        (xdescent_adjusted, ydescent_adjusted),
        width * scale_factor,
        width * scale_factor,  # 使用相同比例的宽和高，保持正方形
        fill=False,
        edgecolor=orig_handle.get_edgecolor(),
        linewidth=orig_handle.get_linewidth(),
    )
    return square


if __name__ == "__main__":
    plot_study_area(90, 270, -72, 71.5)
