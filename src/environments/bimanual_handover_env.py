import os.path
from typing import Any, Dict

import numpy as np

import gymnasium as gym
import yaml
from numpy.typing import NDArray

from src.control.controller import OSCGripperController
from src.control.utils.device import Device
from src.control.utils.enums import GripperState
from src.control.utils.robot import Robot
from src.control.utils.target import Target
from src.environments import BasePandaBimanualEnv
from src.utils.constants import MUJOCO_FRAME_SKIP, MUJOCO_RENDER_FPS
from src.utils.paths import SCENES_DIR, CONTROL_CONFIGS_DIR


class PandaBimanualHandoverEnv(BasePandaBimanualEnv):
    def __init__(
            self,
            scene_file="dual_panda_handover_env.xml",
            control_config_file="dual_panda.yaml",
            robot_name="DualPanda",
            render_mode='human',
            width=480,
            height=360,
            frame_skip=MUJOCO_FRAME_SKIP,
        ):

        super().__init__(scene_file, control_config_file, robot_name, render_mode, width, height, frame_skip)

    @property
    def q_home(self) -> NDArray[np.float64]:
        """
        Return the home position of the robot
        :return: q_home (panda joints + free cuboid joint)
        """
        return np.array(
            [0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04] * 2 + \
            [0.4, -0.4, 0.2, 0, 0, 1, 0],
            dtype=np.float64
        )


if __name__ == "__main__":
    env = PandaBimanualHandoverEnv()
    env.visualize_static()



