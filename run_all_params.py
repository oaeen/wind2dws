from datetime import datetime

import torch.nn as nn
from loguru import logger

from config import Config
from models.model import load_model
from predict_metrics import (
    plot_output_spec_samples,
    process_predict_spec2d,
)
from train import build_prediction_model
from utils.metrics.perf_recorder import *
from utils.plot.metrics_plotter import (
    calculate_freq_cc,
    plot_metrics,
)
from utils.plot.polar_spectrum import *


def run_for_all(eval=False):
    # double scale
    run_config(
        "PointA",
        net="WindNet_ERA5_X",
        global_wind="global_PointA_pool4",
        y_data_source="ERA5",
        eval=eval,
    )
    run_config(
        "PointB",
        net="WindNet_ERA5_X",
        global_wind="global_pool4",
        y_data_source="ERA5",
        eval=eval,
    )
    run_config("CDIP028", net="WindNet_X", global_wind="global_pool4", eval=eval)

    # large scale
    run_config(
        "PointA",
        net="WindNet_ERA5_G",
        global_wind="global_PointA_pool4",
        y_data_source="ERA5",
        eval=eval,
    )
    run_config(
        "PointB",
        net="WindNet_ERA5_G",
        global_wind="global_pool4",
        y_data_source="ERA5",
        eval=eval,
    )
    run_config(
        "CDIP028",
        net="WindNet_G",
        global_wind="global_pool4",
        eval=eval,
    )

    # Local Scale
    run_config(
        "PointA",
        net="WindNet_ERA5_L",
        global_wind="global_PointA_pool4",
        y_data_source="ERA5",
        batch_size=256,
        eval=eval,
    )
    run_config(
        "PointB",
        net="WindNet_ERA5_L",
        global_wind="global_pool4",
        y_data_source="ERA5",
        eval=eval,
    )
    run_config(
        "CDIP028",
        net="WindNet_L",
        global_wind="global_pool4",
        batch_size=256,
        eval=eval,
    )


def set_config(
    y_location,
    net="WindNet_X",
    global_wind="global",
    local_wind_windows=8 * 2,
    global_wind_window=8 * 10,
    y_data_source="IOWAGA",
    batch_size=64,
):
    config = Config()

    config.batch_size = batch_size
    config.network_type = net
    config.y_location = y_location

    config.local_wind_location = y_location
    config.global_wind_location = global_wind

    config.local_wind_window = local_wind_windows
    config.global_wind_window = global_wind_window
    config.input_channels = 2 * config.local_wind_window
    config.train_steps = 1
    config.test_steps = 1
    config.y_data_source = y_data_source

    config.set_model()
    return config


def run_config(
    y_location,
    net="WindNet_X",
    global_wind="global",
    local_wind_windows=8 * 2,
    global_wind_window=8 * 10,
    y_data_source="IOWAGA",
    batch_size=64,
    eval=False,
):
    config = set_config(
        y_location,
        net=net,
        global_wind=global_wind,
        local_wind_windows=local_wind_windows,
        global_wind_window=global_wind_window,
        y_data_source=y_data_source,
        batch_size=batch_size,
    )

    config.set_comment()

    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_handler = logger.add(f"{config.get_log_dir()}/{time}.log")
    logger.info(f"config.comment: {config.comment}")

    if eval is False:
        build_prediction_model(config)

    y_predict, y_true = process_predict_spec2d(config=config)
    y_predict[y_predict < 0] = 0
    np.save(f"{config.get_evaluate_figure_save_dir()}/y_predict.npy", y_predict)
    np.save(f"{config.get_evaluate_figure_save_dir()}/y_true.npy", y_true)
    freq_cc = calculate_freq_cc(y_predict, y_true)
    # convert np array to csv and save
    np.savetxt(
        f"{config.get_evaluate_figure_save_dir()}/freq_cc.csv",
        freq_cc,
        delimiter=",",
    )
    plot_metrics(y_predict[::16], y_true[::16], config)
    plot_output_spec_samples(y_predict, y_true, config=config)

    logger.remove(file_handler)


if __name__ == "__main__":
    run_for_all(eval=True)
