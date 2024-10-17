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
        self.y_location = "W1205N325"  # W1205N325
        self.y_data_desc = "spec_output"  # spec_output

        self.local_wind_location = None
        self.global_wind_location = None

        self.X_spec_scale_bins = "none"  # "freq" or "freq_and_dir" or None
        self.y_scale_desc = "minmax_freq_and_dir"  # "minmax_freq_and_dir" or "minmax_freq" or "log" or "none"
        self.current_freq_index = -1

        self.H_freq_lim = 22
        self.L_freq_lim = 8
        self.global_wind_window = 8 * 10
        self.local_wind_window = 12
        self.train_steps = 7
        self.test_steps = 1
        self.input_channels = None

        # training settings
        self.ensemble_models_num = 5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = 100
        self.batch_size = 80
        self.patience = 7
        self.learning_rate = 1e-4
        self.weight_decay = 0.01
        self.criterion = AdaptiveMSELoss()
        self.optimizer = Lion  # Lion or Adam torch.optim.Adam
        self.output_size = 1

        self.network_type = "WindNet_X"
        self.module_filename = f"models.{self.network_type}"

        self.comment = None

        # Path & Dir
        self.raw_data_dir = "F:/Raw"
        self.processed_data_dir = "F:/Processed"
        self.results_root_dir = f"{os.getcwd()}/results"

    def set_comment(self):
        self.comment = f"{self.network_type}1010_Seele_{self.y_location}_scale_{self.y_scale_desc}_TP{self.global_wind_window}"

    def set_model(self):
        self.module_filename = f"models.{self.network_type}"

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
