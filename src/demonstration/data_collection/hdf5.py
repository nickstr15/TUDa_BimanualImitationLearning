import datetime
import json
import os
from glob import glob

import h5py
import numpy as np

import gymnasium as gym

import src.environments # needed for gym.make to work with custom environments
from src.utils.real_time import RealTimeHandler


def gather_demonstrations_as_hdf5(
    directory : str,
    out_dir : str,
    env_args : dict = None
) -> None:
    """
    Modified version of the sample code from the robosuite library:
    https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/scripts/collect_human_demonstrations.py

    Gathers the demonstrations saved in @directory into a
    single hdf5 file.

    The structure of the hdf5 file is as follows.

    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        env (attribute) - environment name on which demos were collected
        env_args (attribute) - environment arguments as encoded json string

        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration

        demo2 (group)
        ...

    :param directory: Path to the directory containing raw demonstrations.
    :param out_dir: Path to where to store the hdf5 file.
    :param env_args: Additional (fixed) environment arguments as a dictionary.
    """

    # ensure the output directory exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = None  # will get populated at some point

    for ep_directory in os.listdir(directory):

        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []
        success = False

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])
            success = success or dic["successful"]

        if len(states) == 0:
            continue

        # Add only the successful demonstration to dataset
        if success:
            print("Demonstration is successful and has been saved")
            # Delete the last state. This is because when the DataCollector wrapper
            # recorded the states and actions, the states were recorded AFTER playing that action,
            # so we end up with an extra state at the end.
            del states[-1]
            assert len(states) == len(actions)

            num_eps += 1
            ep_data_grp = grp.create_group("demo_{}".format(num_eps))

            # store model xml as an attribute
            xml_path = os.path.join(directory, ep_directory, "model.xml")
            with open(xml_path, "r") as f:
                xml_str = f.read()
            ep_data_grp.attrs["model_file"] = xml_str

            # write datasets for states and actions
            ep_data_grp.create_dataset("states", data=np.array(states))
            ep_data_grp.create_dataset("actions", data=np.array(actions))
        else:
            print("Demonstration is unsuccessful and has NOT been saved")

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.year, now.month, now.day)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["env"] = env_name
    grp.attrs["env_args"] = json.dumps(env_args if env_args is not None else {})

    f.close()

def playback_demonstrations_from_hdf5(
    hdf5_file : str,
    use_actions : bool = False,
    target_real_time : bool = False) -> None:
    """
    Modified version of the sample code from the robosuite library:
    https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/scripts/playback_demonstrations_from_hdf5.py

    Playback the demonstrations saved in the hdf5 file.

    :param hdf5_file: Path to the hdf5 file containing the demonstrations.
    :param use_actions: Whether to use the actions for playback instead of loading the simulation states one by one.
    :param target_real_time: Whether to play back the demonstrations in real time.
    """
    assert os.path.exists(hdf5_file), "File {} does not exist".format(hdf5_file)

    f = h5py.File(hdf5_file, "r")
    env_name = f["data"].attrs["env"]

    env_args = json.loads(f["data"].attrs["env_args"])
    env_args["render_mode"] = "human"

    env = gym.make(
        env_name,
        disable_env_checker=True,
        **env_args
    )

    # list of all demonstrations episodes
    demos = list(f["data"].keys())

    rt = RealTimeHandler(env.unwrapped.render_fps)

    for episode in demos:
        print("Playing back episode: {}".format(episode))

        env.reset()

        # load the flattened mujoco states
        states = f["data/{}/states".format(episode)][()]

        if use_actions:
            #load initial state
            env.unwrapped.set_state_from_flattened(states[0])

            actions = f["data/{}/actions".format(episode)][()]

            rt.reset()
            for j, action in enumerate(actions):
                env.step(action)
                env.render()

                if target_real_time:
                    rt.sleep()

        else:
            # force the sequence of internal mujoco states one by one
            rt.reset()
            for state in states:
                env.unwrapped.set_state_from_flattened(state)
                env.render()

                if target_real_time:
                    rt.sleep()

    env.close()