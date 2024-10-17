import os
import sys
from pathlib import Path

import numpy as np
from loguru import logger

sys.path.append(os.getcwd())
from config import Config
from utils.preprocess.extract.load_data import load_spec, load_wind
from utils.preprocess.prepare.scale_spec import *


def prepare_iowaga_spec(spec_path, save_dir, spec_scale_bins):
    if os.path.isfile(spec_path):
        logger.success(f"load spec from path: {spec_path}")
        spec = np.load(spec_path)
    else:
        logger.success(f"load spec from dir: {spec_path}")
        spec = load_spec(spec_path)

    logger.success(f"load spec data from {spec_path}")
    # 将nan值替换为0
    spec = np.nan_to_num(spec)

    spec_train, spec_test, spec_scaler_dict = scale_spec2d(spec, spec_scale_bins)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    np.save(f"{save_dir}/train.npy", spec_train.astype(np.float32))
    np.save(f"{save_dir}/test.npy", spec_test.astype(np.float32))
    pickle.dump(spec_scaler_dict, open(f"{save_dir}/scaler.pkl", "wb"))
    logger.success(f"save spec & scaler to {save_dir}")


def prepare_iowaga_test_spec(spec_path, save_dir, spec_scale_bins):
    spec_test = np.load(f"{spec_path}/2022.npy")
    spec_test = np.nan_to_num(spec_test)
    scaler = pickle.load(open(f"{save_dir}/scaler.pkl", "rb"))
    spec_test_scaled = scale_spec2d_from_scaler(spec_test, scaler, spec_scale_bins)
    np.save(f"{save_dir}/test.npy", spec_test_scaled.astype(np.float32))


def prepare_iowaga_spec_script():
    # 处理IOWAGA的输出数据 #######################
    y_locations = ["CDIP028", "CDIP045", "CDIP067", "CDIP093", "CDIP107"]
    config.y_data_source = "IOWAGA"
    config.y_data_desc = "spec_output"

    for location in y_locations:
        config.y_location = location
        spec_dir = (
            f"{config.processed_data_dir}/{config.y_data_source}/extract/{location}"
        )
        save_dir = config.get_y_data_dir()
        prepare_iowaga_spec(spec_dir, save_dir, config.y_spec_scale_bins)
        prepare_iowaga_test_spec(spec_dir, save_dir, config.y_spec_scale_bins)


if __name__ == "__main__":
    config = Config()
    prepare_iowaga_spec_script()
