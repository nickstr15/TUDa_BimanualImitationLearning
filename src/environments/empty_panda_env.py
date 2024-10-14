from typing import Dict, Tuple

import numpy as np

import gymnasium as gym
from numpy.typing import NDArray

from src.control.utils.enums import GripperState
from src.control.utils.target import Target
from src.utils.constants import MUJOCO_FRAME_SKIP, DEFAULT_WIDTH, DEFAULT_HEIGHT
from src.environments.core.environment_interface import IEnvironment


class EmptyPandaEnv(IEnvironment):
    """
    Empty environment with two Panda robots.
    """

    def __init__(
        self,
        scene_file : str = "dual_panda_env.xml",
        frame_skip : int = MUJOCO_FRAME_SKIP,
        observation_space : gym.spaces.Space = None,
        control_config_file : str = "dual_panda.yaml",
        robot_name : str = "DualPanda",
        render_mode : str = 'human',
        width : int = DEFAULT_WIDTH,
        height : int = DEFAULT_HEIGHT,
        store_frames : bool = False
    ) -> None:
        """
        Initialize the environment with the specified scene and control config files.

        :param scene_file: filename of to the mujoco scene file in $SCENES_DIR.
               This is the mujoco model_file.
        :param frame_skip: number of simulation steps to skip between each rendered frame
        :param observation_space: observation space of the environment, default is None.
               If None is passed, the observation space is set to the joint positions and velocities
               of the robot joints
        :param control_config_file: filename of the control config file in $CONTROL_CONFIGS_DIR
        :param robot_name: name of the robot in the control config file
        :param render_mode: mujoco render mode
        :param width: width of the render window
        :param height: height of the render window
        :param store_frames: boolean value indicating if **all** the rendered frames should be stored
        """
        super().__init__(
            scene_file, frame_skip, observation_space,
            control_config_file, robot_name,
            render_mode, width, height, store_frames
        )


    @property
    def q_home(self) -> NDArray[np.float64]:
        return np.array(
            [0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04]*2,
            dtype=np.float64
        )


    @property
    def x_home_targets(self) -> Dict[str, Target]:
        targets = {
            "panda_01": Target(),
            "panda_02": Target(),
        }

        targets["panda_01"].set_xyz(np.array([0.55449948,  0.4, 0.68450243]))
        targets["panda_01"].set_quat(np.array([0, 1/np.sqrt(2), 1/np.sqrt(2), 0]))
        targets["panda_01"].set_gripper_state(GripperState.OPEN)

        targets["panda_02"].set_xyz(np.array([0.55449948,  -0.4, 0.68450243]))
        targets["panda_02"].set_quat(np.array([0, 1/np.sqrt(2), 1/np.sqrt(2), 0]))
        targets["panda_02"].set_gripper_state(GripperState.OPEN)
        return targets

    @property
    def _default_free_joint_positions(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Default free joint positions for the environment.
        The empty environment has no free joints and therefor returns an empty dictionary.
        :return:
        """
        return {}

    def _get_obs(self) -> NDArray[np.float64]:
        return np.concatenate([self.data.qpos, self.data.qvel])


if __name__ == "__main__":
    env = EmptyPandaEnv()
    env.visualize_static()
    env.close()


