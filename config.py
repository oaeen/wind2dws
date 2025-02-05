import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from models.lion import Lion
from models.loss import AdaptiveMSELoss


@dataclass
class Config:
    def __init__(self):
        # data settings
        self.y_data_source = "IOWAGA"  # IOWAGA or ERA5
        self.y_location = None
        self.y_data_desc = "spec_output"  # spec_output

        self.local_wind_location = None
        self.large_wind_location = None

        self.y_scale_desc = "minmax_freq_and_dir"
        self.freq_num = 36
        self.out_in = 37

        self.compile_enabled = True
        self.large_wind_window = 8 * 10
        self.local_wind_window = 8 * 3
        self.train_steps = 1
        self.test_steps = 1
        self.input_channels = None

        # training settings
        self.ensemble_models_num = 5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = 100
        self.batch_size = 16
        self.patience = 7
        self.learning_rate = 1e-4
        self.weight_decay = 0.01
        self.criterion = AdaptiveMSELoss()
        self.optimizer = Lion  # Lion or Adam torch.optim.Adam
        self.output_size = 1

        self.model_name = "Wendy"
        self.module_filename = f"models.{self.model_name}"

        self.comment = None

        # Path & Dir
        self.raw_data_dir = "/mnt/f/Raw"
        self.processed_data_dir = "/mnt/e/Processed"
        self.results_root_dir = f"{os.getcwd()}/results"

    def set_comment(self):
        self.comment = f"{self.model_name}_{self.y_location}"

    def set_model(self):
        self.module_filename = f"models.{self.model_name}"

    def get_y_data_dir(self, y_location=None):
        if y_location is None:
            y_location = self.y_location

        target_data_dir = f"{self.processed_data_dir}/{self.y_data_source}/output/{self.y_data_desc}_{self.y_scale_desc}/{y_location}"
        Path(target_data_dir).mkdir(parents=True, exist_ok=True)
        return target_data_dir

    def get_log_dir(self):
        log_dir = f"{self.results_root_dir}/logs/{self.y_location}/{self.comment}"
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        return log_dir

    def get_model_dir(self):
        model_dir = (
            f"{self.results_root_dir}/checkpoints/{self.y_location}/{self.comment}"
        )
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        return model_dir

    def get_samples_figure_save_dir(self):
        samples_save_dir = (
            f"{self.results_root_dir}/samples_output/{self.y_location}/{self.comment}"
        )
        Path(samples_save_dir).mkdir(parents=True, exist_ok=True)
        return samples_save_dir

    def get_evaluate_figure_save_dir(self):
        evaluate_save_dir = (
            f"{self.results_root_dir}/evaluate/{self.y_location}/{self.comment}"
        )
        Path(evaluate_save_dir).mkdir(parents=True, exist_ok=True)
        return evaluate_save_dir
