import gc
import profile
import time

import numpy as np
import torch
import torch.cuda.amp as amp
import torchinfo
from loguru import logger
from scipy.stats import pearsonr
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

from config import Config
from models.model import load_model, save_model
from utils.data_loaders.helper import get_dataloader
from utils.metrics.perf_recorder import *


def train(model, dataloader, optimizer, criterion, config=Config()):
    """
    训练函数, 返回训练损失(MSE)
    """
    model.train()
    train_loss = 0
    scaler = amp.GradScaler()  # 创建梯度缩放器

    for gw, lw, targets in dataloader:
        optimizer.zero_grad()
        gw = gw.to(config.device)
        lw = lw.to(config.device)
        targets = targets.to(config.device)

        # 前向传播（使用 autocast 上下文进行自动混合精度计算）
        with amp.autocast():
            outputs = model(gw, lw)
            loss = criterion(outputs, targets)

        # 反向传播（使用 GradScaler 缩放梯度）
        scaler.scale(loss).backward()
        # 更新模型参数
        scaler.step(optimizer)
        # 更新缩放器
        scaler.update()

        train_loss += loss.item()

    return train_loss / len(dataloader)


@torch.no_grad()
def test(model, dataloader, criterion, config=Config()):
    """
    验证函数, 返回验证损失(MSE),MAE,相关系数
    """
    model.eval()
    val_loss = 0
    all_outputs = []
    all_targets = []

    for gw, lw, targets in dataloader:
        gw = gw.to(config.device)
        lw = lw.to(config.device)
        targets = targets.to(config.device)

        outputs = model(gw, lw)
        loss = criterion(outputs, targets)
        val_loss += loss.item()

        all_outputs.append(outputs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    val_loss /= len(dataloader)
    R, _ = pearsonr(all_targets.flatten(), all_outputs.flatten())
    mse = mean_squared_error(all_targets.flatten(), all_outputs.flatten())

    val_loss_dict = {
        "loss": val_loss,
        "R": R,
        "MSE": mse,
    }

    return val_loss_dict


def build_prediction_model(config=Config()):
    logger.info("Training model...")

    model, best_epoch, best_loss = load_model(config)
    backup_model_and_config(config)

    writer = SummaryWriter(log_dir=config.get_log_dir())
    try:
        global_wind = torch.randn(
            config.batch_size, config.global_wind_window, 2, 72, 60
        )
        local_wind = torch.randn(config.batch_size, config.local_wind_window, 2, 81, 81)
        # writer.add_graph(model, input_to_model=(wind_input))
        torchinfo.summary(
            model, input_size=[global_wind.shape, local_wind.shape], depth=1
        )
    except Exception as e:
        logger.error(f"torchinfo.summary error:{e}")

    train_dataloader, test_dataloader = get_dataloader(config)
    criterion = nn.MSELoss().to(config.device)
    optimizer = config.optimizer(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-7
    )

    no_improvement_count = 0
    criterion_change = False

    for epoch in range(best_epoch, config.epochs):
        t0 = time.time()

        train_loss = train(model, train_dataloader, optimizer, criterion, config)
        val_loss_dict = test(model, test_dataloader, criterion, config)

        t1 = time.time()
        val_loss = val_loss_dict["loss"]
        R = val_loss_dict["R"]
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            no_improvement_count = 0
            save_model(
                model=model, current_epoch=epoch, best_loss=best_loss, config=config
            )
            logger.success(f"test loss improved, saving the current model.")
        else:
            no_improvement_count += 1
            logger.info(f"No improvement for {no_improvement_count} epochs.")

            if no_improvement_count >= config.patience:
                logger.success("Early stopping triggered, stopping the training.")
                break

        report_perf(
            writer=writer,
            current_epoch=epoch,
            run_time=t1 - t0,
            train_loss=train_loss,
            val_loss_dict=val_loss_dict,
            best_val_loss=best_loss,
            best_val_loss_epoch=best_epoch,
            lr=scheduler.get_last_lr()[0],
        )

        if no_improvement_count > 0 and not criterion_change:
            criterion = config.criterion.to(config.device)
            criterion_change = True
            logger.warning("Criterion changed to Relative Loss Next epoch.")
            best_loss = float("inf")

        if val_loss >= best_loss * 50 or np.isnan(R):
            logger.error(
                f"test loss is NaN, try to load the last best model and reduce the learning rate."
            )
            logger.warning(f"Learning rate decayed to {scheduler.get_last_lr()[0]/10}")
            model, best_epoch, best_loss = load_model(config)
            optimizer = config.optimizer(
                model.parameters(),
                lr=scheduler.get_last_lr()[0] / 10,
                weight_decay=config.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-7
            )

    torch.cuda.empty_cache()
    gc.collect()
    logger.success("Training completed.")


if __name__ == "__main__":
    config = Config()
    build_prediction_model(config)
