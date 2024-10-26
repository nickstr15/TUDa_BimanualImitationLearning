import os

from abc import ABC, abstractmethod
from copy import copy
from typing import Any, Dict, Tuple, List, Optional
import time


import mujoco
import numpy as np

import gymnasium as gym
import yaml
from gymnasium import Env
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.mujoco.mujoco_rendering import BaseRender
from gymnasium.envs.registration import EnvSpec
from gymnasium.utils import seeding
from numpy.typing import NDArray
from robosuite.models.base import MujocoXML
from transforms3d.quaternions import qmult
from typing_extensions import override

from src.control.controller import OSCGripperController
from src.control.utils.device import Device
from src.control.utils.enums import GripperState
from src.control.utils.robot import Robot
from src.control.utils.ee_state import EEState
from src.environments.core.action import OSAction
from src.environments.core.enums import ActionMode
from src.environments.core.viewer_config import CamConfig
from src.utils.constants import MUJOCO_FRAME_SKIP, MUJOCO_RENDER_FPS, DEFAULT_WIDTH, DEFAULT_HEIGHT
from src.utils.paths import SCENES_DIR, CONTROL_CONFIGS_DIR

from robosuite.utils.binding_utils import MjSimState


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
            frame_skip : int = MUJOCO_FRAME_SKIP,
            action_mode : ActionMode = ActionMode.ABSOLUTE,
            control_config_file : str = None,
            robot_name : str = None,
            render_mode : str = 'human',
            width : int = DEFAULT_WIDTH,
            height : int = DEFAULT_HEIGHT,
            store_frames : bool = False,
            visualize_targets : bool = False,
            cam_config: CamConfig = CamConfig()
        ) -> None:
        """
        Initialize the environment with the specified scene and control config files.

        ---------------
        Details on the action mode:
        For a single arm (device) the action updates the target

        ABSOLUTE
            - pos_target <- pos_action
            - quat_target <- quat_action
            - grip_target <- grip_action

        RELATIVE
            - pos_target <- pos_action + pos_current
            - quat_target <- quat_action * quat_current
            - grip_target <- grip_action (always override the target)

        :param scene_file: filename of to the mujoco scene file in $SCENES_DIR.
               This is the mujoco model_file.
        :param frame_skip: number of simulation steps to skip between each rendered frame
        :param action_mode: action mode of the environment.
               ABSOLUTE: action input = absolute target position(s)
               RELATIVE: action input = relative target position(s) => target = action + current
        :param control_config_file: filename of the control config file in $CONTROL_CONFIGS_DIR
        :param robot_name: name of the robot in the control config file
        :param render_mode: mujoco render mode
        :param width: width of the render window
        :param height: height of the render window
        :param store_frames: boolean value indicating if **all** the rendered frames
               (of the main viewer) should be stored
        :param visualize_targets: boolean value indicating if the targets should be visualized
        :param cam_config: Configuration for the camera position and orientation
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
            self, full_scene_path, frame_skip, None,
            render_mode=render_mode,
            width=width,
            height=height,
        )

        # setup main viewer
        viewer = self.mujoco_renderer._get_viewer(render_mode)
        viewer._create_overlay = lambda: None
        self.__viewer_setup(viewer, cam_config)

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

        # save the original action space
        self.ctrl_action_space = self.action_space

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

        self._visualize_targets = visualize_targets

        self._targets = self.x_home_targets

        self._action_mode = action_mode

        self._seed = None

        self._register_args(
            action_mode=action_mode
        )

    def _register_args(self, **args) -> None:
        """
        Register all arguments for the environment,
        that are relevant to reinitialize the environment in such a way
        that the same simulation is created.
        :param args: dictionary of arguments
        """
        if not hasattr(self, "_args"):
            self._args = dict()
        self._args.update(args)

    @property
    def args(self) -> Dict:
        """
        Return all important arguments for reinitializing the environment.
        :return: dictionary of arguments
        """
        return self._args if hasattr(self, "_args") else dict()

    @staticmethod
    def __viewer_setup(viewer : BaseRender, cam_config : CamConfig) -> None:
        """
        Set the initial camera position of the main viewer.
        :param viewer: mujoco viewer
        :param cam_config: configuration for the camera
        :return:
        """
        viewer.cam.fixedcamid = -1
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.azimuth = cam_config.azimuth
        viewer.cam.elevation = cam_config.elevation
        viewer.cam.lookat = cam_config.lookat
        viewer.cam.distance = cam_config.distance

    def set_robot_joint_pos(self, joint_pos : np.ndarray) -> None:
        """
        Set the joint positions of the robot.

        :param joint_pos: joint positions
        :return:
        """
        assert len(joint_pos) == self.robot.num_joints_total, "Joint positions must have the same length as the number of joints"
        self.data.qpos[:len(joint_pos)] = joint_pos
        mujoco.mj_forward(self.model, self.data)

    def get_robot_joint_pos(self) -> NDArray[np.float64]:
        """
        Get the joint positions of the robot.
        :return: joint positions
        """
        return self.data.qpos[:self.robot.num_joints_total]

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

        mujoco.mj_forward(self.model, self.data)

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

    def _generate_control(self, targets : Dict[str, EEState]) -> NDArray[np.float64]:
        """
        Generate the control signal (joint torques) for the robot.
        :param targets: target positions and orientations in the world frame
        :return: control array
        """
        controller_output = self.controller.generate(targets)
        ctrl_array = copy(self.data.ctrl)
        for ctrl_idx, ctrl in zip(*controller_output):
            ctrl_array[ctrl_idx] = ctrl
        return ctrl_array

    def get_state(self) -> MjSimState:
        """
        Get MjSimState instance for current state.
        :return: current state
        """
        return MjSimState(
            time=self.data.time,
            qpos=np.copy(self.data.qpos),
            qvel=np.copy(self.data.qvel),
        )

    def set_sim_state(self, state: MjSimState) -> None:
        """
        Set the state of the simulation (RoboSuite version).
        :param state: state to be set
        """
        self.data.time = state.time
        self.set_state(qpos=state.qpos, qvel=state.qvel)

    def set_state_from_flattened(self, state: np.ndarray) -> None:
        """
        Set the state of the simulation from a flattened state.
        :param state: flattened state
        """
        state = MjSimState.from_flattened(state, self)
        self.set_sim_state(state)


    @override
    def step(self, action: OSAction | np.ndarray) -> Tuple[Any, float, bool, bool, Dict]:
        """
        Step function of the environment.
        :param action: dictionary of (relative) targets for the devices
        :return: observation, reward, terminated, truncated, info
        """
        if not isinstance(action, OSAction):
            action = OSAction.from_flattened(action, self._sub_device_names)

        self._update_targets(action)

        self._internal_step()

        obs = self._get_obs()
        info = self._get_info()
        reward = self._get_reward()
        terminated = self._get_terminated()
        truncated = self._get_truncated()
        return obs, reward, terminated, truncated, info

    def set_seed(self, seed):
        self._np_random, self._seed = seeding.np_random(seed)

    @property
    def seed(self) -> int | None:
        return self._seed

    def get_xml(self) -> str:
        """
        Reads a string of the MJCF XML file.

        :return: XML tree read in from file as a string
        """
        xml = MujocoXML(self.fullpath)
        return xml.get_xml()

    def empty_step(self)  -> Tuple[Any, float, bool, bool, Dict]:
        """
        Do step without action input.
        :return: observation, reward, terminated, truncated, info
        """
        self._internal_step()

        obs = self._get_obs()
        info = self._get_info()
        reward = self._get_reward()
        terminated = self._get_terminated()
        truncated = self._get_truncated()
        return obs, reward, terminated, truncated, info

    def _internal_step(self) -> None:
        """
        Execute a step in the environment with targets as goal positions.
        :return: observation, reward, terminated, truncated, info
        """
        ctrl = self._generate_control(self._targets)
        self.do_simulation(ctrl, self.frame_skip)

    def _update_targets(self, action: OSAction) -> None:
        """
        Update the end effector targets.
        :param action: dictionary of targets for the devices
        :return:
        """
        if self._action_mode == ActionMode.ABSOLUTE:
            self._targets = action.get()
        elif self._action_mode == ActionMode.RELATIVE:
            self._targets = self.get_robot_ee_states()
            for device_name, delta in action.items():
                self._targets[device_name].set_xyz(
                    delta.get_xyz() + self._targets[device_name].get_xyz()
                )
                self._targets[device_name].set_quat(
                    qmult(delta.get_quat(), self._targets[device_name].get_quat())
                )
                self._targets[device_name].set_gripper_state(delta.get_gripper_state())

        if self._visualize_targets:
            for device_name, target in self._targets.items():
                target_name = f"target_{device_name}"
                idx = self.get_mocap_idx(target_name)
                if idx > -1: #only if visual target mocap exists
                    self.data.mocap_pos[idx] = target.get_xyz()
                    self.data.mocap_quat[idx] = target.get_quat()

    def visualize_static(self, duration=5) -> None:
        """
        Visualize the robot in the home position for the specified duration.
        :param duration: duration of the visualization
        :return:
        """
        self.reset()

        action = OSAction(self.x_home_targets)
        start_time = time.time()
        while time.time() - start_time < duration:
            self.step(action)
            self.render()

    @property
    def render_fps(self) -> int:
        """
        Return the frames per second of the rendering.
        :return: fps
        """
        return self.metadata['render_fps']

    @override
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        """
        Reset the environment to the initial state.
        Adapted from MujocoEnv.reset() to make use of the options' parameter.
        :param seed:
        :param options: dictionary of additional options for the reset,
            currently only options[randomize] and options[render] are used
        :return:
        """
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        ob = self.reset_model(options)
        info = self._get_reset_info()

        if self.render_mode == "human" and \
           (options.get("render", False) if options is not None else False):
            self.render()
        return ob, info

    @override
    def reset_model(self, options : Optional[dict] = None) -> Dict:
        """
        Reset the model to the initial state.

        :param options: (not used)
        :return: observation of the environment as a dictionary
        """
        self._reset_model(options)
        return self._get_obs()

    def _reset_model(self, options : Optional[dict] = None) -> None:
        """
        Initialize the simulation with the robot at the home position
        and the free joints to the default positions.

        :param options: additional options for the reset (not used)
        :return:
        """
        mujoco.mj_forward(self.model, self.data)
        self.set_robot_joint_pos(self.q_home)

        randomize = options.get("randomize", False) if options is not None else False

        if randomize:
            free_joint_positions = self._get_random_free_joints_quat_pos()
        else:
            free_joint_positions = self._default_free_joints_quat_pos

        for joint_name, pos in free_joint_positions.items():
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
        rendering = self.mujoco_renderer.render(self.render_mode)

        if self._store_frames:
            self._rendered_frames.append(rendering)

        return rendering

    def set_initial_config(self, positions: list) -> None:
        """
        Set the initial configuration of the environment.
        :param positions: list containing the initial configuration for bodies
        :return:
        """
        for body in positions:
            pos_name = body["name"]
            pos = body["pos"]
            quat = body["quat"]
            self.set_free_joint_pos(pos_name, quat, pos)


    def get_robot_ee_states(self) -> Dict[str, EEState]:
        """
        Return the state of the robot as a dictionary.
        :return: dictionary of the robot state
        """
        state = {}
        for device in self.robot.sub_devices:
            state[device.name] = device.get_arm_state()
        return state

    @property
    def action_mode(self):
        return self._action_mode

    @property
    @abstractmethod
    def q_home(self) -> NDArray[np.float64]:
        """
        Return the home position of the robot in the joint space

        :return: joint positions
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def x_home_targets(self) -> Dict[str, EEState]:
        """
        Return the home position of the robot in the world frame as targets
        :return: x_home_target
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def _default_free_joints_quat_pos(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Return the default positions and orientations of the free joints.
        The value of a specific joint is a tuple of the quaternion and the position.
        :return: key-value pairs of free joint names and their position and quaternion as tuples
        """
        raise NotImplementedError

    @abstractmethod
    def _get_random_free_joints_quat_pos(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Return random positions and orientations of the free joints (at the start of an episode).
        The value of a specific joint is a tuple of the quaternion and the position.
        :return: key-value pairs of free joint names and their position and quaternion as tuples
        """
        raise NotImplementedError

    @abstractmethod
    def _get_obs(self) -> Dict:
        """
        Return the observation of the environment as a dictionary.
        Must be compatible with the observation space.
        :return: observation
        """
        raise NotImplementedError

    @abstractmethod
    def _get_info(self) -> Dict:
        """
        Return additional information about the environment state as a dictionary.
        :return: additional state information
        """
        raise NotImplementedError

    @abstractmethod
    def _get_reward(self) -> float:
        """
        Return the reward for the current state.
        :return: reward
        """
        raise NotImplementedError

    @abstractmethod
    def _get_terminated(self) -> bool:
        """
        Return a boolean value indicating if the episode is terminated.
        :return: boolean indicating if the episode is terminated
        """
        raise NotImplementedError

    @abstractmethod
    def _get_truncated(self) -> bool:
        """
        Return a boolean value indicating if the episode is truncated.
        :return: boolean indicating if the episode is truncated
        """
        raise NotImplementedError

    @abstractmethod
    def get_mocap_idx(self, name) -> int:
        """
        Return the id of the mocap object with the specified name.
        This is used to update the target indicators
        :param name: name of the mocap object
        :return: id of the mocap object, -1 if not found
        """
        return -1

    @abstractmethod
    def _check_success(self) -> bool:
        """
        Check if the task is successful.
        :return: boolean value indicating if the task is completed successfully
        """
        raise NotImplementedError


