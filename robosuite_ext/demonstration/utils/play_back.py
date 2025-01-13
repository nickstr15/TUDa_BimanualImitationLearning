import argparse
import json
import os
import random

import h5py
import numpy as np

import robosuite as suite

from robosuite_ext.utils.real_time import RealTimeHandler


def play_back_from_hdf5(
        hdf5_path: str,
        use_actions: bool = False,
        num_episodes: int = None,
        target_rt: bool = False
):
    """
    Play back demonstrations from a hdf5 file.
    :param hdf5_path: path to the hdf5 file containing the demonstrations
    :param use_actions: flag to use the actions for playback instead of loading the simulation states one by one
    :param num_episodes: number of episodes to play back. If None, all episodes are played back.
    :param target_rt: flag to play back the demonstrations in real time
    :return:
    """
    f = h5py.File(hdf5_path, "r")
    env_info = json.loads(f["data"].attrs["env_info"])

    env = suite.make(
        **env_info,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
    )

    rt = RealTimeHandler(env.control_freq)

    # list of all demonstrations episodes
    demos = list(f["data"].keys())
    num_episodes = num_episodes if num_episodes is not None else len(demos)
    num_episodes = min(num_episodes, len(demos))

    demos_to_play = random.sample(demos, num_episodes)

    for ep in demos_to_play:
        # read the model xml, using the metadata stored in the attribute for this episode
        model_xml = f["data/{}".format(ep)].attrs["model_file"]

        env.reset()
        xml = env.edit_model_xml(model_xml)
        env.reset_from_xml_string(xml)
        env.sim.reset()
        env.viewer.set_camera(0)

        # load the flattened mujoco states
        states = f["data/{}/states".format(ep)][()]

        if use_actions:

            # load the initial state
            env.sim.set_state_from_flattened(states[0])
            env.sim.forward()

            # load the actions and play them back open-loop
            actions = np.array(f["data/{}/actions".format(ep)][()])
            num_actions = actions.shape[0]

            rt.reset()
            for j, action in enumerate(actions):
                env.step(action)

                if j < num_actions - 1:
                    # ensure that the actions deterministically lead to the same recorded states
                    state_playback = env.sim.get_state().flatten()
                    if not np.all(np.equal(states[j + 1], state_playback)):
                        err = np.linalg.norm(states[j + 1] - state_playback)
                        print(f"[warning] playback diverged by {err:.2f} for ep {ep} at step {j}")

                if target_rt:
                    rt.sleep()

        else:

            # force the sequence of internal mujoco states one by one
            rt.reset()
            for state in states:
                env.sim.set_state_from_flattened(state)
                env.sim.forward()
                if env.renderer == "mjviewer":
                    env.viewer.update()
                env.render()

                if target_rt:
                    rt.sleep()
    f.close()