import os
import time

import wandb
import yaml
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
import robosuite as suite

from src.imitation_learning.core.dataset import HDF5Dataset
from src.utils.paths import path_completion, DEMOS_DIR, TRAINED_MODELS_DIR, LOG_DIR
from src.utils.wandb import WANDB_API_KEY


class ExperimentBase:
    def __init__(self, config_path: str):
        """
        Base class for all experiments.
        :param config_path: path to the config file
        """

        self._config = self._load_config(config_path)
        self._setup_seed()
        self._setup_device()
        self._setup_paths()
        self._setup_dataset()
        self._setup_env()
        self._setup_algorithm()
        self._setup_logging()

    @staticmethod
    def _load_config(config_path: str):
        """
        Load the configuration file.
        :param config_path: path to the config file
        :return: configuration dictionary
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config

    def _setup_seed(self):
        """
        Set up the seed for reproducibility.
        """
        seed = self._config["seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _setup_device(self):
        """
        Set up the device for training.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _setup_paths(self):
        """
        Set up the paths for saving models.
        """
        self._model_out_dir = path_completion(self._config["model_out_dir"], TRAINED_MODELS_DIR)
        self._log_dir = path_completion(self._config["log_dir"], LOG_DIR)

        # add timestamp in format YYYY-MM-DD_HH-MM-SS
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self._model_out_dir = os.path.join(self._model_out_dir, timestamp)
        self._log_dir = os.path.join(self._log_dir, timestamp)

        os.makedirs(self._model_out_dir, exist_ok=False)
        os.makedirs(self._log_dir, exist_ok=False)

    def _setup_dataset(self):
        """
        Set up the dataset.
        """
        self._dataset = HDF5Dataset(
            path_completion(self._config["dataset"]["hdf5_path"], DEMOS_DIR),
            observation_length=self._config["environment"]["obs_length"],
            action_length=self._config["environment"]["action_length"],
            uses_state_obs=self._config["dataset"]["uses_state_obs"],
            uses_image_obs=self._config["dataset"]["uses_image_obs"],
            specific_obs_keys=self._config["dataset"]["specific_obs_keys"]
        )

        training, evaluation = self._dataset.get_splits(self._config["dataset"]["split_ratio"])

        self._train_loader = DataLoader(training, batch_size=self._config["dataset"]["batch_size"], shuffle=True)
        self._eval_loader = DataLoader(evaluation, batch_size=self._config["dataset"]["batch_size"], shuffle=True)

        self._input_size = self._dataset.input_size
        self._output_size = self._dataset.output_size

    def _setup_env(self):
        """
        Set up the environment.
        """
        self._env = self._dataset.env

        assert self._env.__name__ == self._config["env"]["name"], "Environment name mismatch"


    def _setup_algorithm(self):
        """
        Set up the algorithm.
        """
        raise NotImplementedError

    def _setup_logging(self):
        """
        Set up the logging.
        """

        self._log_wandb = self._config["log_wandb"]

        if self._log_wandb:
            assert WANDB_API_KEY is not None, "WANDB_API_KEY is not set"

            os.environ["WANDB_API_KEY"] = WANDB_API_KEY

