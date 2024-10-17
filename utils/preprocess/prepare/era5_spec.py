import os
import sys
from pathlib import Path

import numpy as np
from loguru import logger

sys.path.append(os.getcwd())
from config import Config
from utils.preprocess.extract.load_data import load_spec, load_wind
from utils.preprocess.prepare.scale_spec import *
from utils.plot.metrics_plotter import plot_hist2d
from utils.plot.polar_spectrum import plot_spec
from utils.metrics.spectra_info import get_spec_desc


def prepare_era5_spec(spec_path, save_dir, spec_scale_bins):
    if os.path.isfile(spec_path):
        logger.success(f"load spec from path: {spec_path}")
        spec = np.load(spec_path)
    else:
        logger.success(f"load spec from dir: {spec_path}")
        spec = load_spec(spec_path)

    logger.success(f"load spec data from {spec_path}")
    # 将nan值替换为0
    spec = np.nan_to_num(spec)
    print(f"spec.shape: {spec.shape}")

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    spec_train, spec_test, spec_scaler_dict = scale_spec2d(spec, spec_scale_bins)
    pickle.dump(spec_scaler_dict, open(f"{save_dir}/scaler.pkl", "wb"))

    np.save(f"{save_dir}/train.npy", spec_train.astype(np.float32))
    np.save(f"{save_dir}/test.npy", spec_test.astype(np.float32))

    logger.success(f"save spec & scaler to {save_dir}")


if __name__ == "__main__":
    config = Config()
    config.y_scale_desc = "none"
    config.y_data_source = "ERA5"
    era5_points = {
        "PointA": (15, 115),  # 15°N, 115°W
        "PointB": (5, -120),  # 5°N, 120°W
    }
    for key, value in era5_points.items():
        config.y_location = key
        target_lat, target_lon = value
        logger.info(
            f"preparing spec for {config.y_location} at {target_lat}, {target_lon}"
        )

        save_dir = f"{config.processed_data_dir}/{config.y_data_source}/output/{config.y_data_desc}_{config.y_scale_desc}/{config.y_location}"
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        prepare_era5_spec(
            f"{config.processed_data_dir}/ERA5/extract/spec_extract_{config.y_location}",
            save_dir,
            config.y_scale_desc,
        )
