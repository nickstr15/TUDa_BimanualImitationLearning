import os

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import time


import mujoco
import numpy as np

import gymnasium as gym
import yaml
from gymnasium.envs.mujoco import MujocoEnv
from numpy.typing import NDArray
from typing_extensions import override

from src.control.controller import OSCGripperController
from src.control.utils.device import Device
from src.control.utils.robot import Robot
from src.control.utils.target import Target
from src.utils.constants import MUJOCO_FRAME_SKIP, MUJOCO_RENDER_FPS, DEFAULT_WIDTH, DEFAULT_HEIGHT
from src.utils.paths import SCENES_DIR, CONTROL_CONFIGS_DIR


class IEnvironment(MujocoEnv, ABC):
    """
    Interface for robot manipulation tasks in Mujoco.
    """

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
            scene_file : str = None,
            frame_skip: int = MUJOCO_FRAME_SKIP,
            observation_space: gym.spaces.Space = None,
            control_config_file : str = None,
            robot_name : str = None,
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

        assert scene_file is not None, "Scene file must be specified!"
        assert control_config_file is not None, "Control config file must be specified!"
        assert robot_name is not None, "Robot name must be specified!"

        full_scene_path = os.path.join(SCENES_DIR, scene_file)
        assert os.path.exists(full_scene_path), \
            f"Scene file {full_scene_path} does not exist!"

        full_control_config_path = os.path.join(CONTROL_CONFIGS_DIR, control_config_file)
        assert os.path.exists(full_control_config_path), \
            f"Control config file {full_control_config_path} does not exist!"

        MujocoEnv.__init__(
            self, full_scene_path, frame_skip, observation_space,
            render_mode=render_mode,
            width=width,
            height=height,
        )

        if observation_space is None:
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

        self._sub_device_names = [dev.name for dev in sub_devices]
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

        self._store_frames = store_frames
        if self._store_frames:
            assert render_mode == 'rgb_array', "Frames can only be stored when render_mode is 'rgb_array'"
        self._rendered_frames = [] if self._store_frames else None


    @staticmethod
    def _viewer_setup(viewer) -> None:
        """
        Set the initial camera position of the viewer.
        :param viewer: mujoco viewer
        :return:
        """
        viewer.cam.azimuth = 190
        viewer.cam.elevation = -20
        viewer.cam.lookat[0] = 0.0
        viewer.cam.lookat[1] = 0.0
        viewer.cam.lookat[2] = 0.1
        viewer.cam.distance = 2.5

    def set_robot_joint_pos(self, joint_pos : np.ndarray) -> None:
        """
        Set the joint positions of the robot.

        :param joint_pos: joint positions
        :return:
        """
        assert len(joint_pos) == self.robot.num_joints_total, "Joint positions must have the same length as the number of joints"
        self.data.qpos[:len(joint_pos)] = joint_pos

    def set_free_joint_pos(self, free_joint_name : str , quat : np.ndarray = None, pos : np.ndarray = None):
        """
        Set the position of the free joint.

        :param free_joint_name: name of the free joint
        :param quat: orientation of the free joint as a quaternion
        :param pos: position of the free joint (x, y, z)
        :return:
        """
        jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, free_joint_name)
        offset = self.model.jnt_qposadr[jnt_id]

        if quat is not None:
            quat_idxs = np.arange(offset + 3, offset + 7)  # Quaternion indices
            self.data.qpos[quat_idxs] = quat

        if pos is not None:
            pos_idxs = np.arange(offset, offset + 3)  # Position indices
            self.data.qpos[pos_idxs] = pos

    @staticmethod
    def __get_devices(mj_model, mj_data, cfg, use_sim=True) -> Any:
        """
        Create the devices and robots from the configuration file.
        :param mj_model: the mujoco model
        :param mj_data: the mujoco data
        :param cfg: the configuration file as a dictionary
        :param use_sim: boolean value indicating if the simulation is being used
        :return: devices
        """
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
        """
        Get the robot with the specified name.
        :param robot_name: name of the desired robot
        :return: instance of the robot
        """
        for device in self._devices:
            if type(device) == Robot:
                if device.name == robot_name:
                    return device

    def __get_controller_config(self, name) -> dict:
        """
        Get the controller configuration with the specified name.
        :param name: name of the controller
        :return: configuration dictionary
        """
        ctrlr_conf = self._control_config["controller_configs"]
        for entry in ctrlr_conf:
            if entry["name"] == name:
                return entry

    def _generate_control(self, targets : Dict[str, Target], relative_targets : bool = False) -> NDArray[np.float64]:
        """
        Generate the control signal (joint torques) for the robot.
        :param targets: target positions and orientations in the world frame
        :param relative_targets: boolean value indicating if the targets are relative to the current position
        :return: control array
        """
        controller_output = self.controller.generate(targets, relative_targets)
        ctrl_array = np.zeros_like(self.data.ctrl)
        for ctrl_idx, ctrl in zip(*controller_output):
            ctrl_array[ctrl_idx] = ctrl
        return ctrl_array

    def visualize_static(self, duration=5) -> None:
        """
        Visualize the robot in the home position for the specified duration.
        :param duration: duration of the visualization
        :return:
        """
        self.reset()

        targets = self.x_home_targets
        start_time = time.time()
        while time.time() - start_time < duration:
            ctrl = self._generate_control(targets)
            self.do_simulation(ctrl, self.frame_skip)

            self.render()

    @property
    def render_fps(self) -> int:
        """
        Return the frames per second of the rendering.
        :return: fps
        """
        return self.metadata['render_fps']

    def reset_model(self) -> NDArray[np.float64]:
        """
        Reset the model to the initial state.
        :return: observation
        """
        self._reset_model()
        return self._get_obs()

    def _reset_model(self) -> None:
        """
        Initialize the simulation with the robot at the home position
        and the free joints to the default positions.

        :return:
        """
        mujoco.mj_forward(self.model, self.data)
        self.set_robot_joint_pos(self.q_home)
        for joint_name, pos in self._default_free_joint_positions.items():
            self.set_free_joint_pos(joint_name, *pos)

    def get_mujoco_renders(self):
        """
        Return the rendered frames.
        :return: all rendered frames
        """
        return self._rendered_frames

    @override
    def render(self) -> Any:
        """
        Render the environment.
        :return: the rendering output of the mujoco renderer
        """
        rendering = self.mujoco_renderer.render(
            self.render_mode, self.camera_id, self.camera_name
        )

        if self._store_frames:
            self._rendered_frames.append(rendering)

        return rendering

    @property
    @abstractmethod
    def q_home(self) -> NDArray[np.float64]:
        """
        Return the home position of the robot in the joint space

        :return: joint positions
        """
        pass

    @property
    @abstractmethod
    def x_home_targets(self) -> Dict[str, Target]:
        """
        Return the home position of the robot in the world frame as targets
        :return: x_home_target
        """
        pass

    @abstractmethod
    def _get_obs(self) -> NDArray[np.float64]:
        """
        Return the observation of the environment.
        Must be compatible with the observation space.
        :return: observation
        """
        pass

    @property
    @abstractmethod
    def _default_free_joint_positions(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Return the default positions of the free joints.
        The value of a specific joint is a tuple of the quaternion and the position.
        :return: key-value pairs of free joint names and their positions
        """
        pass