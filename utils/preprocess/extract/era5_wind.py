import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from loguru import logger
from tqdm import tqdm
from skimage.measure import block_reduce

sys.path.append(os.getcwd())

from config import Config
from utils.preprocess.prepare.blocker import get_target_area_blocker


def extract_target_area_wind_data(
    filepath,
    lat_range,
    lon_range,
    time_interval=3,
    pool_kernel_size=1,
):
    """
    提取指定经纬度范围内的风速数据
    """
    data = xr.open_dataset(filepath, engine="netcdf4")

    # 假设你的数据中经度是从0到360表示的，将西经125°到110°转换为对应的正值
    # 如果你的数据中经度是从-180到180表示的，则直接使用-125到-110
    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range

    if lon_min < 0:
        lon_min = 360 + lon_min
    if lon_max < 0:
        lon_max = 360 + lon_max

    # 选择特定的经纬度区域
    target_data = data.sel(
        longitude=slice(lon_min, lon_max), latitude=slice(lat_min, lat_max)
    )
    # 提取u,v风速数据
    u_wind = target_data["u10"].values
    v_wind = target_data["v10"].values

    # 按3小时间隔抽取数据
    u_wind = u_wind[::time_interval]
    v_wind = v_wind[::time_interval]
    if pool_kernel_size != 1:
        block_size = (1, pool_kernel_size, pool_kernel_size)
        # 进行平均池化
        u_wind = block_reduce(u_wind, block_size, np.mean)
        v_wind = block_reduce(v_wind, block_size, np.mean)
    u_wind = np.expand_dims(u_wind, axis=1)
    v_wind = np.expand_dims(v_wind, axis=1)
    wind_info = np.concatenate([u_wind, v_wind], axis=1)
    print(wind_info.shape)
    return wind_info


def extract_wind_data(data_dir, save_dir, lat_range, lon_range, pool_kernel_size=1):
    """
    提取指定经纬度范围内的风速数据
    """
    Path.mkdir(Path(save_dir), parents=True, exist_ok=True)

    for year in tqdm(range(1993, 2023), desc="extracting..."):
        if os.path.exists(f"{data_dir}/{year}.npy"):
            logger.success(f"File {year}.npy exists.")
            continue

        logger.info(f"Extracting wind data from year {year}")

        wind_speed_list = []
        for month in range(1, 13):
            filepath = f"{data_dir}/{year}/ERA5_wind_{year}{month:02d}.nc"
            if not os.path.exists(filepath):
                logger.error(f"File {filepath} not exists.")
                continue
            logger.info(f"Extracting data from {filepath}")
            wind_speed = extract_target_area_wind_data(
                filepath, lat_range, lon_range, pool_kernel_size=pool_kernel_size
            )
            wind_speed_list.append(wind_speed)

        wind_speed = np.concatenate(wind_speed_list, axis=0)
        logger.info(f"wind speed shape: {wind_speed.shape}")
        save_path = f"{save_dir}/{year}.npy"
        wind_speed = wind_speed.astype(np.float32)
        np.save(save_path, wind_speed)


if __name__ == "__main__":
    config = Config()
    data_dir = f"{config.raw_data_dir}/ERA5/Wind_Global/data"
    save_dir = f"{config.processed_data_dir}/ERA5/extract/wind_extracted"

    era5_points = {
        "PointA": (15, 115),  # 15°N, 115°E
        "PointB": (5, -120),  # 5°N, 120°W
    }

    for key, value in era5_points.items():
        loc_name = key
        target_lat, target_lon = value
        logger.info(
            f"Extracting wind data for {loc_name} at {target_lat}, {target_lon}"
        )
        save_dir = f"{config.processed_data_dir}/ERA5/extract/wind_extracted_{loc_name}"
        lat_range = [target_lat + 20, target_lat - 20]
        lon_range = [target_lon - 20, target_lon + 20]
        extract_wind_data(
            data_dir,
            save_dir,
            lat_range,
            lon_range,
        )

    # global
    # global_lat_range = [71.5, -72]
    # global_lon_range = [150.5, -90]

    # global_PointB
    # global_lat_range = [71.5, -72]
    # global_lon_range = [90.5, -150]

    # save_dir = (
    #     f"{config.processed_data_dir}/ERA5/extract/wind_extracted_global_PointB_pool4"
    # )
    # extract_wind_data(
    #     data_dir, save_dir, global_lat_range, global_lon_range, pool_kernel_size=4
    # )

    buyo_locations = {
        "CDIP028": (34, -118.5),
    }
    for key, value in buyo_locations.items():
        loc_name = key
        target_lat, target_lon = value
        logger.info(
            f"Extracting wind data for {loc_name} at {target_lat}, {target_lon}"
        )
        save_dir = f"{config.processed_data_dir}/ERA5/extract/wind_extracted_{loc_name}"
        lat_range = [target_lat + 20, target_lat - 20]
        lon_range = [target_lon - 20, target_lon + 20]
        extract_wind_data(data_dir, save_dir, lat_range, lon_range)
