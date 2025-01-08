import collections
import json
from typing import OrderedDict

import h5py
import numpy as np
import robosuite as suite
import torch

from torch.utils.data import Dataset
from torch.utils.data import random_split

import src.environments

class HDF5Dataset(Dataset):
    def __init__(self,
             hdf5_path,
             observation_length : int = 1,
             action_length : int = 1,
             uses_state_obs : bool = True,
             uses_image_obs : bool = False,
             specific_obs_keys : list = None
         ):
        """
        Dataset class for loading demonstrations from a hdf5 file.
        :param hdf5_path: path to the hdf5 file containing the demonstrations
        :param observation_length: number of observations to be used for each state
        :param action_length: number of actions to be used for each state
        :param uses_state_obs: flag to use state observations
        :param uses_image_obs: flag to use image observations
        :param specific_obs_keys: list of specific observation keys to use,
            this is not compatible with state or image observations being used
        :return:
        """
        assert observation_length > 0, "observation_length must be greater than 0"
        assert action_length > 0, "action_length must be greater than 0"

        self.observation_length = observation_length
        self.action_length = action_length

        with h5py.File(hdf5_path, "r") as f:
            env_info = json.loads(f["data"].attrs["env_info"])

            # list of all demonstrations episodes
            self.episode_keys = list(f["data"].keys())
            self.episode_lens = [len(f["data/{}".format(ep)]['states']) for ep in self.episode_keys]

            self.states = [f["data/{}/states".format(ep)][()] for ep in self.episode_keys]
            self.actions = [f["data/{}/actions".format(ep)][()] for ep in self.episode_keys]

            f.close()

        print("env_info")
        print(env_info)

        self.env = suite.make(
            **env_info,
            has_renderer=True,
            has_offscreen_renderer=False,
            ignore_done=True,
            use_camera_obs=False,
        )

        #print available observations
        print("-" * 50)
        print("Available observations:")
        available_keys = self.env._get_observations().keys()
        for key in available_keys:
            print(key)
        print("-" * 50)

        self._specific_obs_keys = specific_obs_keys if specific_obs_keys is not None else []
        if uses_state_obs:
            for key in available_keys:
                if key.endswith("_pos") or key.endswith("_quat"): # or key.endswith("_qpos"):
                    self._specific_obs_keys.extend(key)

        if uses_image_obs:
            raise NotImplementedError("Image observations are not yet supported")

        # make list of specific observation keys unique
        self._specific_obs_keys = list(set(self._specific_obs_keys))

    @property
    def obs_keys(self):
        """
        Get the used observation keys
        :return: list of keys
        """
        return self._specific_obs_keys

    def __len__(self):
        return sum(self.episode_lens)

    @property
    def input_sizes(self):
        """
        Get the size of the input
        :return: size of the input
        """
        sizes = {}
        for k in self._specific_obs_keys:
            sizes[k] = self[0]["observations"][k].shape
        return sizes

    @property
    def output_size(self):
        """
        Get the size of the output
        :return: size of the output
        """
        return self[0]["actions"].shape

    def __getitem__(self, idx):
        """
        Get the idx-th state in the dataset
        :param idx: index for the item
        :return: state, action
        """
        # find the episode that idx belongs to
        ep_idx = 0
        while idx >= self.episode_lens[ep_idx]:
            idx -= self.episode_lens[ep_idx]
            ep_idx += 1

        # get the observation_length states before idx
        start_idx = max(0, idx - self.observation_length + 1)
        states = self.states[ep_idx][start_idx:idx+1]
        # pad with first state if necessary
        if len(states) < self.observation_length:
            states = [states[0]] * (self.observation_length - len(states)) + states

        # get the action_length actions after idx
        end_idx = min(idx + self.action_length, self.episode_lens[ep_idx])
        actions = self.actions[ep_idx][idx:end_idx]
        # pad with last action if necessary
        if len(actions) < self.action_length:
            actions = actions + [actions[-1]] * (self.action_length - len(actions))

        # collect observations
        observations = collections.OrderedDict()
        for state in states:
            self.env.sim.set_state_from_flattened(state)
            self.env.sim.forward()
            self.env.update_state()

            observation = self.env._get_observations(force_update=True)
            for k, v in self._extract_observation(observation).items():
                if k not in observations:
                    observations[k] = np.expand_dims(v, axis=0)
                else:
                    observations[k] = np.concatenate([observations[k], np.expand_dims(v, axis=0)])

        actions = torch.from_numpy(actions).type(torch.float32)

        for k, v in observations.items():
            observations[k] = torch.from_numpy(v).type(torch.float32)

        return {
            "observations": observations,
            "actions": actions
        }

    def _extract_observation(self, observation: OrderedDict) -> OrderedDict:
        """
        Extract the relevant observations from the OrderedDict
        :param observation: OrderedDict containing the observations
        :return: extracted observations
        """
        extracted_observation = collections.OrderedDict()
        for key in self._specific_obs_keys:
            extracted_observation[key] = observation[key]

        return extracted_observation

    # get indexes for train and test rows
    def get_splits(self, split_ratio: float = 0.8):
        train_size = int(split_ratio * len(self))
        test_size = len(self) - train_size
        # calculate the split
        return random_split(self, [train_size, test_size])