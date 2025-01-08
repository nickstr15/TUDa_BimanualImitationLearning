import os
import time
import logging
from abc import ABC, abstractmethod

import wandb
import yaml
import torch
import random
import numpy as np
import json
from torch.utils.data import DataLoader

from src.imitation_learning.core.dataset import HDF5Dataset
from src.imitation_learning.policies.policy import PolicyBase
from src.utils.paths import path_completion, DEMOS_DIR, TRAINED_MODELS_DIR, LOG_DIR, IL_PATH
from src.utils.wandb import WANDB_API_KEY

class ExperimentBase(ABC):
    """
    Base class for all experiments.

    Requires the following methods to be implemented:
        - _setup_policy
    """
    def __init__(self, config_path: str):
        """
        Base class for all experiments.

        Requires the following methods to be implemented:
            - _setup_policy

        :param config_path: path to the config file
        """
        self._config = self._load_config(config_path)
        self._setup_seed()
        self._setup_device()
        self._setup_dataset()
        self._setup_env()

        self.policy = self._setup_policy()

    @abstractmethod
    def _setup_policy(self) -> PolicyBase:
        """
        Set up the policy.
        """
        raise NotImplementedError

    def run(self):
        """
        Run the experiment.
        """
        self._setup_logging_and_paths()

        self.policy.train_loop(
            self._train_loader,
            self._eval_loader,
            num_epochs=self._config["training"]["num_epochs"],
            num_epochs_eval=self._config["training"]["num_epochs_eval"],
            num_episodes_eval=self._config["training"]["num_episodes_eval"],
            model_out_dir=self._model_out_dir,
            logger=self._logger,
            log_wandb=self._log_wandb,
            eval_env=self._env,
            obs_keys=self._dataset.obs_keys
        )

    def load_and_visualize_policy(self, model_path: str, num_episodes: int = 5):
        """
        Load a model and visualize the policy.
        :param model_path: path to the model file
        :param num_episodes: number of episodes to visualize
        """
        self._setup_logging_and_paths(False)

        model_path = path_completion(model_path, TRAINED_MODELS_DIR)

        self.policy.load(model_path)
        self.policy.visualize(self._env, obs_keys=self._dataset.obs_keys, num_episodes=num_episodes)

    @staticmethod
    def _load_config(config_path: str):
        """
        Load the configuration file.
        :param config_path: path to the config file
        :return: configuration dictionary
        """
        config_path = path_completion(config_path, os.path.join(IL_PATH, "configs"))
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

        self._input_sizes = self._dataset.input_sizes
        self._output_size = self._dataset.output_size

    def _setup_env(self):
        """
        Set up the environment.
        """
        self._env = self._dataset.env

        #disable rendering depending on flag
        self._env.has_renderer = self._config["environment"]["visualize_eval_runs"]
        if not self._env.has_renderer:
            self._env.viewer = None

        assert self._env.__class__.__name__ == self._config["environment"]["name"], "Environment name mismatch"

    def _setup_logging_and_paths(self, training: bool = True):
        """
        Set up the logging.
        """
        if training:
            self._model_out_dir = path_completion(self._config["model_out_dir"], TRAINED_MODELS_DIR)
            self._log_dir = path_completion(self._config["log_dir"], LOG_DIR)

            # add timestamp in format YYYY-MM-DD_HH-MM-SS
            self._timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            self._model_out_dir = os.path.join(self._model_out_dir, self._timestamp)
            self._log_dir = os.path.join(self._log_dir, self._timestamp)
            self._log_path = os.path.join(self._log_dir, "log.log")

            os.makedirs(self._model_out_dir, exist_ok=False)
            os.makedirs(self._log_dir, exist_ok=False)

            self._log_wandb = self._config["wandb"]["log"]

            if self._log_wandb:
                assert WANDB_API_KEY is not None, "WANDB_API_KEY is not set"
                os.environ["WANDB_API_KEY"] = WANDB_API_KEY

                wandb.init(
                    project=self._config["wandb"]["project"],
                    config=self._config,
                    dir=self._log_dir,
                    group=self._env.__class__.__name__,
                    name=self._env.__class__.__name__ + "_" + self._timestamp
                )

            # Configure logging
            self._logger = logging.getLogger("Experiment Logger")  # Use __name__ for module-specific logging
            self._logger.setLevel(logging.INFO)  # Set the logging level

            # Create a file handler
            file_handler = logging.FileHandler(self._log_path)
            file_handler.setLevel(logging.INFO)  # Set level for file logging

            # Create a console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)  # Set level for console logging

            # Define a common formatter
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

            # Attach the formatter to the handlers
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            # Add the handlers to the logger
            self._logger.addHandler(file_handler)
            self._logger.addHandler(console_handler)

            config_string = json.dumps(self._config, indent=4)
            self._logger.info(f"Config:\n{config_string}")

        # disable robosuite logging
        logging.getLogger("robosuite_logs").setLevel(logging.ERROR)



