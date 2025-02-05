from datetime import datetime

from loguru import logger

from config import Config
from predict_metrics import (
    plot_output_spec_samples,
    process_predict_spec2d,
)
from train import build_model
from utils.metrics.perf_recorder import *
from utils.plot.metrics_plotter import (
    calculate_freq_cc,
    calculate_freq_bias,
    plot_metrics,
)
from utils.plot.polar_spectrum import *


def run_all_models():
    locations = [
        ("PointA", 30, "ERA5", "large"),
        ("PointB", 30, "ERA5", "large_PB"),
        ("CDIP028", 36, "IOWAGA", "large"),
    ]
    for model_name in ["Wendy", "Wendy_Large", "Wendy_Local"]:
        configure_model(model_name, locations=locations)

    locations = [
        ("PointC", 30, "ERA5", "large_Full"),
    ]
    for model_name in ["Wendy_Full", "Wendy_Large_Full", "Wendy_Local"]:
        configure_model(model_name, locations=locations)


def eval_all_models():
    locations = [
        # ("PointA", 30, "ERA5", "large"),
        # ("PointB", 30, "ERA5", "large_PB"),
        # ("PointC", 30, "ERA5", "large"),
        ("CDIP028", 36, "IOWAGA", "large"),
    ]
    for model_name in [
        "Wendy",
        # "Wendy_Large",
        # "Wendy_Local",
    ]:  # "Wendy", "Wendy_Large", "Wendy_Local"
        configure_model(
            model_name, run_id=1, locations=locations, eval=True, eval_finetune=True
        )


def configure_model(model_name, locations, eval=False, eval_finetune=True):
    for y_location, freq_num, y_data_source, large_wind in locations:
        create_model(
            model_name=model_name,
            y_location=y_location,
            y_data_source=y_data_source,
            freq_num=freq_num,
            large_wind=large_wind,
            eval=eval,
            eval_finetune=eval_finetune,
        )


def create_model(
    y_location,
    y_data_source="IOWAGA",
    freq_num=36,
    model_name="Wendy",
    large_wind="large",
    eval=False,
    eval_finetune=False,
):
    config = Config()
    config.y_location = y_location
    config.y_data_source = y_data_source
    config.freq_num = freq_num
    config.model_name = model_name
    config.local_wind_location = y_location
    config.large_wind_location = large_wind

    config.set_model()
    config.comment = f"{config.model_name}_{config.y_location}"

    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_handler = logger.add(f"{config.get_log_dir()}/{time}.log")
    logger.info(f"config.comment: {config.comment}")

    if eval is False:
        build_model(config)

    y_pre_file = f"{config.get_evaluate_figure_save_dir()}/y_predict.npy"
    y_true_file = f"{config.get_evaluate_figure_save_dir()}/y_true.npy"
    if os.path.exists(y_pre_file) and os.path.exists(y_true_file):
        y_predict = np.load(y_pre_file)
        y_true = np.load(y_true_file)
    else:
        y_predict, y_true = process_predict_spec2d(
            load_finetune_model=eval_finetune, config=config
        )
        y_predict[y_predict < 0] = 0
        np.save(f"{config.get_evaluate_figure_save_dir()}/y_predict.npy", y_predict)
        np.save(f"{config.get_evaluate_figure_save_dir()}/y_true.npy", y_true)

    freq_cc = calculate_freq_cc(y_predict, y_true)
    np.savetxt(
        f"{config.get_evaluate_figure_save_dir()}/freq_cc.csv",
        freq_cc,
        delimiter=",",
    )
    freq_bias = calculate_freq_bias(y_true, y_predict)
    np.savetxt(
        f"{config.get_evaluate_figure_save_dir()}/freq_bias.csv",
        freq_bias,
        delimiter=",",
    )
    plot_metrics(y_predict[::16], y_true[::16], config)
    plot_output_spec_samples(y_predict, y_true, config=config)

    logger.remove(file_handler)


if __name__ == "__main__":

    run_all_models()
    # eval_all_models()
