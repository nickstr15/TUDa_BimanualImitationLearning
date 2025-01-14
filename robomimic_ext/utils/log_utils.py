import os
import sys
import time

from robomimic.utils.log_utils import (
    DataLogger as OriginalDataLogger,
    PrintLogger as OriginalPrintLogger,
    log_warning
)


class DataLogger(OriginalDataLogger):
    """
    Logging class to log metrics to tensorboard/wandb and/or retrieve running statistics about logged data.

    Modified version of the original DataLogger class to better handle wandb logging.
    """
    def __init__(self, log_dir, config, log_tb=True, log_wandb=False):
        """
        Args:
            log_dir (str): base path to store logs
            config (Namespace): configuration dictionary
            log_tb (bool): whether to use tensorboard logging
            log_wandb (bool): whether to use wandb logging
        """
        self._tb_logger = None
        self._wandb_logger = None
        self._data = dict()  # store all the scalar data logged so far

        if log_tb:
            from tensorboardX import SummaryWriter
            self._tb_logger = SummaryWriter(os.path.join(log_dir, 'tb'))

        if log_wandb:
            import wandb
            from utils.wandb import WANDB_API_KEY

            # set up wandb api key if specified in macros
            if WANDB_API_KEY is not None:
                os.environ["WANDB_API_KEY"] = WANDB_API_KEY

            # attempt to set up wandb 10 times. If unsuccessful after these trials, don't use wandb
            num_attempts = 10
            for attempt in range(num_attempts):
                try:
                    # set up wandb
                    self._wandb_logger = wandb

                    self._wandb_logger.init(
                        project=config.experiment.logging.wandb_proj_name,
                        name=config.experiment.name,
                        dir=log_dir,
                        mode="offline" if attempt == num_attempts - 1 else "online",
                    )

                    # set up info for identifying experiment
                    wandb_config = {k: v for (k, v) in config.meta.items() if k not in ["hp_keys", "hp_values"]}
                    for (k, v) in zip(config.meta["hp_keys"], config.meta["hp_values"]):
                        wandb_config[k] = v
                    if "algo" not in wandb_config:
                        wandb_config["algo"] = config.algo_name
                    self._wandb_logger.config.update(wandb_config)

                    break
                except Exception as e:
                    log_warning("wandb initialization error (attempt #{}): {}".format(attempt + 1, e))
                    self._wandb_logger = None
                    time.sleep(10) #wait for 10 seconds before retrying


class PrintLogger(OriginalPrintLogger):
    """
    This class redirects print statements to both console and a file.

    Extension to the original PrintLogger class -> adds isatty() check to avoid printing to file
    """

    def isatty(self):
        """
        Check if the file descriptor is an interactive terminal.
        """
        return self.terminal.isatty()
