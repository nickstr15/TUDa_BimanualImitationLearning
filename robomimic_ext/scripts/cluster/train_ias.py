"""
Basic experiment for IL on the cluster.
If running on the cluster, this experiment should be called from launch_train_ias.py.

This setup disables all rendering and does not save videos.
The setup always tries to use the GPU if available.
"""
import json
import os.path
import traceback
import time
import datetime

import robomimic.utils.torch_utils as torch_utils
from experiment_launcher import single_experiment, run_experiment
from robomimic.config import config_factory

from utils.paths import path_completion, RM_EXP_CONFIG_DIR, RM_DEFAULT_OUTPUT_DIR, DATASET_DIR

from robomimic_ext.scripts.train import train

# noinspection DuplicatedCode
def prep_training_run_cluster(
    config_path: str = None,
    seed: int = -1,
    time_str: str = None,
    debug: bool = False
):
    if config_path is not None:
        full_config_path = path_completion(config_path, RM_EXP_CONFIG_DIR)
        ext_cfg = json.load(open(full_config_path, 'r'))
        config = config_factory(ext_cfg["algo_name"])
        # update config with external json - this will throw errors if
        # the external config has keys not present in the base algo config
        with config.values_unlocked():
            config.update(ext_cfg)
            config.train.data = path_completion(config.train.data, DATASET_DIR)

            # always try to use GPU if available
            config.train.cuda = True

            # disable rendering and video saving
            config.experiment.render = False
            config.experiment.render_video = False
    else:
        raise ValueError("Must provide a config file to run training.")

    with config.values_unlocked():
        if config.train.output_dir.lower() == "DEFAULT".lower():
            config.train.output_dir = RM_DEFAULT_OUTPUT_DIR
        if seed >= 0:
            config.train.seed = seed
            config.train.output_dir = os.path.join(config.train.output_dir, "seed", "{0:0=2d}".format(seed))
        if time_str is not None:
            config.experiment.name = "{}_{}".format(config.experiment.name, time_str)

    # get torch device
    device = torch_utils.get_torch_device(try_to_use_cuda=config.train.cuda)

    # maybe modify config for debugging purposes
    if debug:
        # shrink length of training to test whether this run is likely to crash
        config.unlock()
        config.lock_keys()

        #add debug flag to experiment name
        config.experiment.name = "{}_DEBUG".format(config.experiment.name)

        # train and validate (if enabled) for 3 gradient steps, for 2 epochs
        config.experiment.epoch_every_n_steps = 3
        config.experiment.validation_epoch_every_n_steps = 3
        config.train.num_epochs = 2

        # if rollouts are enabled, try 2 rollouts at end of each epoch, with 10 environment steps
        config.experiment.rollout.rate = 1
        config.experiment.rollout.n = 2
        config.experiment.rollout.horizon = 10


    # lock config to prevent further modifications and ensure missing keys raise errors
    config.lock()

    return config, device

@single_experiment
def experiment(
    config_path: str,
    debug: bool = False,
    time_float: float = time.time(),
    #######################################
    # MANDATORY
    seed: int = 0,
    results_dir: str = "logs",

    #######################################
    # OPTIONAL
    # accept unknown arguments
    **kwargs
):
    time_str = datetime.datetime.fromtimestamp(time_float).strftime('%Y%m%d%H%M%S')

    config, device = prep_training_run_cluster(config_path, seed, time_str, debug)

    print(config.train.output_dir)
    # catch error during training and print it
    res_str = "Finished run successfully!"
    try:
        train(config, device=device)
    except Exception as e:
        res_str = "Run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
    print(res_str)


