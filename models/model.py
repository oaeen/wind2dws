import importlib
import os
import platform
from datetime import datetime
from pathlib import Path

import torch
from loguru import logger
from packaging import version
import numpy as np
import sys

sys.path.append(os.getcwd())

from config import Config


def get_model_info(config=Config()):
    module = importlib.import_module(config.module_filename)
    Net = getattr(module, config.network_type)

    model = Net(
        input_channels=config.input_channels,
        output_size=config.output_size,
    ).to(config.device)

    return model


def load_model(config=Config()):
    """
    加载已训练好的当前频率 或 前一个频率的模型，加快训练速度
    """
    loaded_model = False
    best_epoch = 0
    best_loss = float("inf")

    model = get_model_info(config)
    checkpoint_model_path = (
        f"{config.get_model_dir()}/best_model_freq{config.current_freq_index}.pt"
    )
    loaded_model, best_epoch, best_loss = load_checkpoint(model, checkpoint_model_path)

    if loaded_model is False:
        logger.warning(
            f"Model file {checkpoint_model_path} does not exist, training from scratch!"
        )

    return model, best_epoch, best_loss


def load_checkpoint(model, checkpoint_model_path):
    if os.path.exists(checkpoint_model_path):
        logger.success(f"Loaded model from checkpoint: {checkpoint_model_path}")
        checkpoint = torch.load(checkpoint_model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        return True, checkpoint["epoch"], checkpoint["best_loss"]
    else:
        return False, 0, float("inf")


def save_model(model, current_epoch=0, best_loss=float("inf"), config=Config()):
    """
    保存模型
    """

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": current_epoch,
        "best_loss": best_loss,
    }

    torch.save(
        checkpoint,
        f"{config.get_model_dir()}/best_model_freq{config.current_freq_index}.pt",
    )


def export_onnx(config=Config()):
    model, best_epoch, best_loss = load_model(config)
    model.eval()

    spec_input = torch.rand(1, config.global_wind_window * 1, 36, 24).to(config.device)
    onnx_filename = f"{os.getcwd()}/export/{config.comment}.onnx"
    torch.onnx.export(model, spec_input, onnx_filename, verbose=True)
    logger.success(f"ONNX 模型已成功保存到 {onnx_filename}")


def calculate_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    param_size_MB = param_size / (1024**2)
    return param_size_MB
