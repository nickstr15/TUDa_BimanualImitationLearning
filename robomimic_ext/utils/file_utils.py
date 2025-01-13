import os
import json
import h5py
from robomimic.envs.env_base import EnvType


def get_env_metadata_from_dataset(dataset_path):
    """
    Retrieves env metadata from dataset.
    This is different from the robomimic version to add the missing data.

    Args:
        dataset_path (str): path to dataset

    Returns:
        env_meta (dict): environment metadata. Contains 3 keys:

            :`'env_name'`: name of environment
            :`'type'`: type of environment, should be a value in EB.EnvType
            :`'env_kwargs'`: dictionary of keyword arguments to pass to environment constructor
    """
    dataset_path = os.path.expanduser(dataset_path)
    f = h5py.File(dataset_path, "r")
    env_kwargs = json.loads(f["data"].attrs["env_info"])

    env_meta = dict()
    env_meta["env_name"] = env_kwargs.pop("env_name")
    env_meta["env_kwargs"] = env_kwargs
    env_meta["type"] = EnvType.ROBOSUITE_TYPE #HARD CODED. BECAUSE THIS IS MISSING

    f.close()

    return env_meta