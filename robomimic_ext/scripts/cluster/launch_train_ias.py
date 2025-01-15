"""
Main entry point for launching training experiments on the cluster.
Can also be used to run multiple experiments locally in parallel.

This script disables all rendering and does not save videos.
The script always tries to use the GPU if available.

Args:
    config, c (str): path to a config json that will be used to override the default settings.
        If omitted, default settings are used. This is the preferred way to run experiments.

    n_seeds, n (int): number of seeds to run, defaults to @DEFAULT_N_SEEDS

    memory_single_job, m (int): memory to allocate per job, defaults to @DEFAULT_MEMORY_SINGLE_JOB

    debug, d (bool): set this flag to run a quick training run for debugging purposes

Example usage (from the root directory of the project):

$> python -m robomimic_ext.scripts.cluster.launch_train_ias -c config.json

with config.json being a config file in the robomimic_ext/exp_configs directory.


EXPERIENCES FOR MEMORY ALLOCATION (DEFAULT_MEMORY_SINGLE_JOB):
- BC: 3000
- BC_TRANSFORMER: 4000
- DIFFUSION_UNET: 4000
"""
import time
import argparse

from experiment_launcher import Launcher, is_local

IS_LOCAL = is_local()
N_EXPS_IN_PARALLEL = 1
N_CORES = N_EXPS_IN_PARALLEL
DEFAULT_MEMORY_SINGLE_JOB = 2000
CONDA_ENV = "bil-py310"
PARTITION = "stud"
DEFAULT_N_SEEDS = 5

def launch_experiments(
    config_path: str,
    n_seeds: int = DEFAULT_N_SEEDS,
    memory_single_job: int = DEFAULT_MEMORY_SINGLE_JOB,
    debug: bool = False
):
    """
    Launches training experiments on the cluster.
    :param config_path: path to the config file
    :param n_seeds: number of seeds to run
    :param memory_single_job: memory to allocate per job
    :param debug: whether to run in debug mode
    """

    launcher = Launcher(
        exp_name="bil",
        exp_file="robomimic_ext.scripts.cluster.train_ias",
        n_seeds=n_seeds,
        n_exps_in_parallel=N_EXPS_IN_PARALLEL,
        n_cores=N_EXPS_IN_PARALLEL,
        memory_per_core= N_EXPS_IN_PARALLEL * memory_single_job // N_CORES,
        days=0,
        hours=23,
        minutes=59,
        seconds=0,
        partition=PARTITION,
        conda_env=CONDA_ENV,
        use_timestamp=True,
        compact_dirs=False,
    )

    launcher.add_experiment(
        config_path=config_path,
        debug=debug,
        time_float = time.time(),
    )

    launcher.run(IS_LOCAL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        required=True,
        help="(MANDATORY) Path to a config json that will be used to override the default settings.",
    )

    parser.add_argument(
        "--n_seeds", "-n",
        type=int,
        default=DEFAULT_N_SEEDS,
        help=f"Number of seeds to run, defaults to {DEFAULT_N_SEEDS}"
    )

    parser.add_argument(
        "--memory_single_job", "-m",
        type=int,
        default=DEFAULT_MEMORY_SINGLE_JOB,
        help=f"Memory to allocate per job, defaults to {DEFAULT_MEMORY_SINGLE_JOB}"
    )

    parser.add_argument(
        "--debug", "-d",
        action='store_true',
        help="Set this flag to run a quick training run for debugging purposes"
    )
    args = parser.parse_args()

    launch_experiments(
        args.config,
        args.n_seeds,
        args.memory_single_job,
        args.debug
    )

