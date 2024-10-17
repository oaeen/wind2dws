from datetime import datetime
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os


import sys

sys.path.append(os.getcwd())

from config import Config


class StudyArea:
    def __init__(self) -> None:
        # 添加实心圆点和注释
        self.buyo_locations = {
            "Point C": (-118.6, 33.9),
        }


point_color = "#e83929"
rect_color = "#0eb83a"


def plot_study_area(lon_start, lon_end, lat_start, lat_end, study_area=StudyArea()):
    # 从数据集中提取你需要的区域数据
    config = Config()

    ds = xr.open_dataset(
        f"{config.raw_data_dir}/ETOPO/ETOPO_2022_v1_30s_N90W180_bed.nc"
    )

    subset = ds.sel(lon=slice(lon_start, lon_end), lat=slice(lat_start, lat_end))

    max_depth = 5000
    cust_cmap_val = 0.4
    cust_cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_cmap", [(0, "#ffffff"), (cust_cmap_val, "#007bbb"), (1.0, "#223a70")]
    )
    lat_formatter = FuncFormatter(format_lat_tick)
    lon_formatter = FuncFormatter(format_lon_tick)

    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_extent([lon_start, lon_end, lat_start, lat_end])  # 修改了起始经度为-122

    topo_contour = ax.contourf(
        subset.lon,
        subset.lat,
        -subset.z,
        cmap=cust_cmap,
        transform=ccrs.PlateCarree(),
        levels=np.linspace(0, max_depth, 200),
    )

    cbar = plt.colorbar(
        topo_contour,
        orientation="horizontal",
        pad=0.09,
        ax=ax,
        ticks=np.arange(0, max_depth + 1, 1000),
        aspect=25,
        shrink=0.57,
        anchor=(0.5, 0.6),
    )
    cbar.set_label("Depth (m)", size=8)
    cbar.ax.tick_params(labelsize=8, direction="in")

    ax.add_feature(cfeature.LAND, color="lightgrey")
    ax.add_feature(cfeature.COASTLINE)

    ax.set_xticks(
        np.arange(int(lon_start + 0.5), int(lon_end + 0.5), 2), crs=ccrs.PlateCarree()
    )
    ax.set_yticks(
        np.arange(int(lat_start + 0.5), int(lat_end) + 0.4, 1), crs=ccrs.PlateCarree()
    )
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude", fontsize=8)
    ax.tick_params(axis="both", labelsize=8, direction="in")
    ax.set_title(
        "Southern California Bight",
        fontdict={"fontsize": 13, "fontweight": "bold", "fontname": "Calibri"},
        loc="center",
    )

    for place, (lon, lat) in study_area.buyo_locations.items():
        ax.scatter(
            lon,
            lat,
            color=point_color,
            s=18,
            zorder=10,
            edgecolors="black",
            linewidths=0.5,
        )  # 添加实心圆点
        ax.annotate(
            place,
            (lon, lat),
            textcoords="offset points",
            xytext=(-19, -6),
            ha="center",
            fontsize=9,
        )
        ax.annotate(
            "(028)",
            (lon, lat),
            textcoords="offset points",
            xytext=(-19, -14),
            ha="center",
            fontsize=9,
        )

    time_stamp = datetime.now().strftime("%m%d-%H%M%S")
    plt.savefig(
        f"{os.getcwd()}/results/study_SCB_{time_stamp}.png",
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

    value = f"{value:.0f}" if value.is_integer() else f"{value:.1f}"
    return f"{value}°{n_s}"


def format_lon_tick(value, pos):
    print(f"value: {value}")
    if value < 0:
        value = value + 360

    e_w = ""
    if abs(value - 180) > 1e-6:
        if value < 180:
            e_w = "E"
        if value > 180:
            value = 360 - value
            e_w = "W"

    value = f"{value:.0f}" if value.is_integer() else f"{value:.1f}"
    return f"{value}°{e_w}"


if __name__ == "__main__":

    plot_study_area(-122.5, -115.5, 29.5, 36.5)
