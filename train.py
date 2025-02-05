import gc
import profile
import time

import numpy as np
import torch
import torchinfo
from loguru import logger
from scipy.stats import pearsonr
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from config import Config
from models.model import load_model, save_model
from utils.data_loaders.helper import get_dataloader
from utils.metrics.perf_recorder import *


def train(model, dataloader, optimizer, criterion, config=Config()):
    model.train()
    train_loss = 0
    scaler = torch.amp.GradScaler(device="cuda")
    for large, local, targets in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        large = large.to(config.device)
        local = local.to(config.device)
        targets = targets.to(config.device)

        with torch.autocast(device_type="cuda"):
            outputs = model(large, local)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

    return train_loss / len(dataloader)


@torch.no_grad()
def test(model, dataloader, criterion, config=Config()):
    model.eval()
    val_loss = 0

    for large, local, targets in tqdm(dataloader, desc="Validating"):
        large = large.to(config.device)
        local = local.to(config.device)
        targets = targets.to(config.device)
        outputs = model(large, local)
        loss = criterion(outputs, targets)
        val_loss += loss.item()

    val_loss /= len(dataloader)

    val_loss_dict = {
        "loss": val_loss,
        "R": 0.0,  # placeholder
    }

    return val_loss_dict


def build_model(config=Config()):
    logger.info("Training model...")
    finetune = False

    model_filename = "best_model_finetune.pt"
    if os.path.exists(f"{config.get_model_dir()}/{model_filename}"):
        logger.warning(f"Model file {model_filename} exists, loading the model.")
        model, best_epoch, best_loss = load_model(model_filename, config=config)
        criterion = config.criterion.to(config.device)
        finetune = True
    else:
        logger.warning(
            f"Model file {model_filename} does not exist, training from scratch."
        )
        model, best_epoch, best_loss = load_model(config=config)
        criterion = torch.nn.MSELoss().to(config.device)

    backup_model_and_config(config)

    writer = SummaryWriter(log_dir=config.get_log_dir())
    train_dataloader, test_dataloader = get_dataloader(config)
    optimizer = config.optimizer(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=1, min_lr=1e-7
    )
    scheduler.step(best_loss)
    no_improvement_count = 0

    for epoch in range(best_epoch + 1, config.epochs):
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
                model=model,
                current_epoch=epoch,
                best_loss=best_loss,
                best=True,
                finetune=finetune,
                config=config,
            )
            logger.success(f"test loss improved, saving the current model.")
        else:
            no_improvement_count += 1
            logger.info(f"No improvement for {no_improvement_count} epochs.")

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

        if (no_improvement_count == 7) and not finetune:
            criterion = config.criterion.to(config.device)
            finetune = True
            logger.warning("Criterion changed to Adaptive Loss Next epoch.")
            best_loss = float("inf")
            no_improvement_count = 0
            optimizer = config.optimizer(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.2, patience=2, min_lr=1e-7
            )
            scheduler.step(best_loss)

        if no_improvement_count > config.patience:
            logger.success("Early stopping triggered, stopping the training.")
            break

        if val_loss >= best_loss * 50 or np.isnan(R):
            logger.error(
                f"test loss is NaN, try to load the last best model and reduce the learning rate."
            )
            logger.warning(f"Learning rate decayed to {scheduler.get_last_lr()[0]/10}")
            model, best_epoch, best_loss = load_model(config=config)
            optimizer = config.optimizer(
                model.parameters(),
                lr=scheduler.get_last_lr()[0] / 10,
                weight_decay=config.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.2, patience=2, min_lr=1e-7
            )

    torch.cuda.empty_cache()
    gc.collect()
    logger.success("Training completed.")


if __name__ == "__main__":
    config = Config()
    build_model(config)
