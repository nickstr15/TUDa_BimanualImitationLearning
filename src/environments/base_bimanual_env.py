import os.path
import time
from abc import ABC, abstractmethod
from typing import Tuple, List, Any, Dict

import numpy as np

import mujoco

import gymnasium as gym
import yaml
from gymnasium import Space
from gymnasium.envs.mujoco import MujocoEnv
from numpy import dtype, ndarray
from numpy.typing import NDArray

from src.control.controller import OSCGripperController
from src.control.utils.device import Device
from src.control.utils.enums import GripperState
from src.control.utils.robot import Robot
from src.control.utils.target import Target
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
            robot_name="DualPanda",
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
            observation_space=None, #is set directly after __init__
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

        # set control
        with open(full_control_config_path, 'r') as f:
            self._control_config = yaml.safe_load(f)

        self._devices = self.__get_devices(self.model, self.data, self._control_config)
        self.robot = self.__get_robot(robot_name=robot_name)

        sub_devices = []
        for device in self._devices:
            if type(device) == Device:
                sub_devices += [device]
            elif type(device) == Robot:
                sub_devices += device.sub_devices


        sub_device_configs = [
            (device.name, self.__get_controller_config(device.controller_type))
            for device in sub_devices
        ]
        nullspace_config = self.__get_controller_config("nullspace")
        admittance_gain = self.__get_controller_config("admittance")["gain"]

        self.controller = OSCGripperController(
            robot=self.robot,
            input_device_configs=sub_device_configs,
            nullspace_config=nullspace_config,
            use_g=True,
            admittance_gain=admittance_gain,
        )

    @staticmethod
    def _viewer_setup(viewer) -> None:
        viewer.cam.azimuth = 190
        viewer.cam.elevation = -20
        viewer.cam.lookat[0] = 0.0
        viewer.cam.lookat[1] = 0.0
        viewer.cam.lookat[2] = 0.1
        viewer.cam.distance = 2.5

    @property
    def q_home(self) -> NDArray[np.float64]:
        """
        Return the home position of the robot
        :return: q_home
        """
        return np.array(
            [0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04]*2,
            dtype=np.float64
        )

    @property
    def x_home(self) -> Any:
        """
        Return the home position of the robot
        :return: x_home
        """
        return [
            0.55449948,  0.4, 0.68450243, 0, 1/np.sqrt(2), 1/np.sqrt(2), 0, GripperState.OPEN,
            0.55449948, -0.4, 0.68450243, 0, 1/np.sqrt(2), 1/np.sqrt(2), 0, GripperState.OPEN
        ]

    @property
    def x_home_targets(self) -> Dict[str, Target]:
        """
        Return the home position of the target
        :return: x_home_target
        """
        targets = {
            "panda_01": Target(),
            "panda_02": Target(),
        }

        targets["panda_01"].set_xyz(self.x_home[:3])
        targets["panda_01"].set_quat(self.x_home[3:7])
        targets["panda_01"].set_gripper_state(self.x_home[7])

        targets["panda_02"].set_xyz(self.x_home[8:11])
        targets["panda_02"].set_quat(self.x_home[11:15])
        targets["panda_02"].set_gripper_state(self.x_home[15])
        return targets

    @staticmethod
    def __get_devices(mj_model, mj_data, cfg, use_sim=True) -> Any:
        all_devices = np.array(
            [Device(dev, mj_model, mj_data, use_sim) for dev in cfg["devices"]]
        )
        robots = np.array([])
        all_robot_device_idxs = np.array([], dtype=np.int32)
        for robot_cfg in cfg["robots"]:
            robot_device_idxs = robot_cfg["device_ids"]
            all_robot_device_idxs = np.hstack([all_robot_device_idxs, robot_device_idxs])
            robot = Robot(
                all_devices[robot_device_idxs], robot_cfg["name"], mj_model, mj_data, use_sim
            )
            robots = np.append(robots, robot)

        all_idxs = np.arange(len(all_devices))
        keep_idxs = np.setdiff1d(all_idxs, all_robot_device_idxs)
        devices = np.hstack([all_devices[keep_idxs], robots])
        return devices

    def __get_robot(self, robot_name) -> Robot:
        for device in self._devices:
            if type(device) == Robot:
                if device.name == robot_name:
                    return device

    def __get_controller_config(self, name) -> dict:
        ctrlr_conf = self._control_config["controller_configs"]
        for entry in ctrlr_conf:
            if entry["name"] == name:
                return entry

    def _generate_control(self, targets : Dict[str, Target]) -> NDArray[np.float64]:
        controller_output = self.controller.generate_absolute(targets)
        ctrl_array = np.zeros_like(self.data.ctrl)
        for ctrl_idx, ctrl in zip(*controller_output):
            ctrl_array[ctrl_idx] = ctrl
        return ctrl_array

    def visualize(self, duration = 10) -> None:
        # move robot to home position
        self.set_state(qpos=self.q_home, qvel=np.zeros_like(self.data.qvel))

        targets = self.x_home_targets

        targets["panda_01"].set_xyz(targets["panda_01"].get_xyz() + np.array([0.1, 0, 0]))
        targets["panda_02"].set_xyz(targets["panda_02"].get_xyz() + np.array([-0.1, 0, 0]))

        start_time = time.time()
        while time.time() - start_time < duration:
            ctrl = self._generate_control(targets)
            self.do_simulation(ctrl, self.frame_skip)

            self.render()


if __name__ == "__main__":
    env = BasePandaBimanualEnv()
    env.visualize()
