import importlib
import os

import torch
from loguru import logger
import sys
from torch.profiler import profile, record_function, ProfilerActivity


sys.path.append(os.getcwd())

from config import Config


def get_model_info(config=Config()):
    module = importlib.import_module(config.module_filename)
    Net = getattr(module, config.model_name)

    model = Net(freq_num=config.freq_num).to(config.device)

    if config.compile_enabled:
        logger.info(f"Compiling model using mode default")
        model = torch.compile(model)

    return model


def load_model(filename="best_model.pt", config=Config()):
    """
    加载已训练好的当前频率 或 前一个频率的模型，加快训练速度
    """
    best_epoch = 0
    best_loss = float("inf")

    model = get_model_info(config)

    checkpoint_model_path = f"{config.get_model_dir()}/{filename}"
    model, best_epoch, best_loss = load_checkpoint(model, checkpoint_model_path, config)

    if best_epoch == 0:
        logger.warning(
            f"Model file {checkpoint_model_path} does not exist, training from scratch!"
        )

    return model, best_epoch, best_loss


def load_checkpoint(model, checkpoint_model_path, config=Config()):
    if os.path.exists(checkpoint_model_path):
        checkpoint = torch.load(checkpoint_model_path, weights_only=False)
        logger.success(f"Loaded model from checkpoint: {checkpoint_model_path}")
        model_sd = checkpoint["model_state_dict"]
        if config.compile_enabled is False:
            model_sd = clean_compile_cpt_name(model_sd)
        model.load_state_dict(model_sd)

        return model, checkpoint["epoch"], checkpoint["best_loss"]
    else:
        return model, 0, float("inf")


def clean_compile_cpt_name(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("_orig_mod.", "")
        new_state_dict[new_key] = v
    return new_state_dict


def save_model(
    model,
    current_epoch=0,
    best_loss=float("inf"),
    best=True,
    finetune=False,
    config=Config(),
):
    """
    保存模型
    """

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": current_epoch,
        "best_loss": best_loss,
    }

    filename = f"model_epoch_{current_epoch}"
    if best is True and finetune is False:
        filename = "best_model"
    elif best is True and finetune is True:
        filename = "best_model_finetune"

    torch.save(
        checkpoint,
        f"{config.get_model_dir()}/{filename}.pt",
    )


def calculate_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    param_size_MB = param_size / (1024**2)
    return param_size_MB


def torch_profile(model, large_wind, local_wind):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.xpu.is_available():
        device = "xpu"
    else:
        print(
            "Neither CUDA nor XPU devices are available to demonstrate profiling on acceleration devices"
        )
        import sys

        sys.exit(0)

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA, ProfilerActivity.XPU]
    sort_by_keyword = device + "_time_total"

    with profile(activities=activities, record_shapes=True) as prof:
        with record_function("model_inference"):
            model(large_wind, local_wind)

    print(prof.key_averages().table(sort_by=sort_by_keyword, row_limit=10))
