import os.path
import time
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

import mujoco

import gymnasium as gym
import yaml
from gymnasium import Space
from gymnasium.envs.mujoco import MujocoEnv
from numpy import dtype
from numpy.typing import NDArray

from src.control.utils.device import Device
from src.utils.constants import MUJOCO_FRAME_SKIP, MUJOCO_RENDER_FPS
from src.utils.paths import ENVIRONMENTS_DIR, SCENES_DIR, CONTROL_CONFIGS_DIR


class BasePandaBimanualEnv(MujocoEnv):
    metadata = {
        'render_modes': [
            'human',
            'rgb_array',
            'depth_array'
        ],
        'render_fps' : MUJOCO_RENDER_FPS,
    }

    def __init__(
            self,
            scene_file="dual_panda_env.xml",
            control_config_file="dual_panda.yaml",
            render_mode='human',
            width=480,
            height=360,
            frame_skip=MUJOCO_FRAME_SKIP,
        ):

        full_scene_path = os.path.join(SCENES_DIR, scene_file)
        assert os.path.exists(full_scene_path), \
            f"Scene file {full_scene_path} does not exist!"

        full_control_config_path = os.path.join(CONTROL_CONFIGS_DIR, control_config_file)
        assert os.path.exists(full_control_config_path), \
            f"Control config file {full_control_config_path} does not exist!"

        MujocoEnv.__init__(
            self,
            model_path=full_scene_path,
            frame_skip=frame_skip,
            observation_space=None,
            render_mode=render_mode,
            width=width,
            height=height,
        )

        # set observation space
        obs_size = self.data.qpos.size + self.data.qvel.size
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )
        # setup viewer
        viewer = self.mujoco_renderer._get_viewer(render_mode)
        viewer._create_overlay = lambda: None
        self._viewer_setup(viewer)

        # set action space
        self.ctrl_action_space = self.action_space
        # self.action_space = ... #TODO (depends on controller)

        # set control
        with open(full_control_config_path, 'r') as f:
            self._control_config = yaml.safe_load(f)

    def _viewer_setup(self, viewer) -> None:
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -45
        viewer.cam.lookat[0] = 0.0
        viewer.cam.lookat[1] = 0.0
        viewer.cam.lookat[2] = 0.0
        viewer.cam.distance = 2.5

    def reset_model(self) -> NDArray[np.float64]:
        # nothing to do, just return the observation
        return self._get_obs()

    def _get_obs(self) -> NDArray[np.float64]:
        return np.concatenate([
            self.data.qpos.flat,
            self.data.qvel.flat
        ])

    def visualize(self, duration = 10) -> None:
        self.reset()
        self.set_state(self.home[0], np.zeros_like(self.data.qvel))
        start_time = time.time()
        while time.time() - start_time < duration:
            ctrl = self.home[1]
            self.do_simulation(ctrl, self.frame_skip)
            self.render()

    @property
    def home(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Return the home position of the robot at index 0 and the corresponding control signal at index 1
        :return: (q_home, ctrl_home)
        """
        return (
            np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04]*2, dtype=np.float64),
            np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 255]*2, dtype=np.float64)
        )

if __name__ == "__main__":
    env = BasePandaBimanualEnv()
    env.visualize()
