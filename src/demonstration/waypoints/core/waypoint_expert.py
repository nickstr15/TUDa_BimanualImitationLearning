import json
import os.path
import shutil
import time
from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np
import yaml
import re

from robosuite.controllers.parts.arm import OperationalSpaceController
from robosuite.environments.manipulation.two_arm_env import TwoArmEnv
from robosuite.utils.transform_utils import quat2axisangle, quat_multiply, euler2mat, mat2quat, axisangle2quat, \
    quat_inverse, quat2mat, mat2euler
from robosuite.wrappers import DataCollectionWrapper

from src.demonstration.waypoints.core.waypoint import Waypoint, DEFAULT_MUST_REACH, DEFAULT_MIN_DURATION, \
    DEFAULT_MAX_DURATION, DEFAULT_POSITION_TOLERANCE, DEFAULT_ORIENTATION_TOLERANCE
from src.demonstration.waypoints.utils.null_orientation import get_two_arm_null_orientation
from src.utils.clipping import clip_translation, clip_quat_by_axisangle
from src.utils.paths import WAYPOINTS_DIR
from src.utils.real_time import RealTimeHandler
from src.utils.robot_states import TwoArmEEState, EEState
from src.utils.robot_targets import GripperTarget
from src.demonstration.utils.gather_demonstrations import gather_demonstrations_as_hdf5
from src.wrappers.target_visualization_wrapper import TargetVisualizationWrapper


class TwoArmWaypointExpertBase(ABC):
    """
    Class for an expert agent that acts in the environment
    by following a predefined trajectory of waypoints.

    Only applicable for two-arm environments with OSC for each arm.
    """
    def __init__(
        self,
        environment : TwoArmEnv,
        waypoints_file : str,
    ) -> None:
        """
        Constructor for the WaypointExpert class.
        :param environment: Environment in which the expert agent acts. MUST use use_object_obs=True.
        :param waypoints_file: File containing the waypoints n $WAYPOINTS_DIR
        """
        full_waypoints_path = os.path.join(WAYPOINTS_DIR, waypoints_file)
        assert os.path.isfile(full_waypoints_path), f"Waypoints file {full_waypoints_path} not found"
        self._env = environment

        # get action mode + input/output ranges
        self._evaluate_controller_setup()

        self._null_quat_right, self._null_quat_left = get_two_arm_null_orientation([r.name for r in self._env.robots])

        with open(full_waypoints_path, 'r') as f:
            self._waypoint_cfg = yaml.safe_load(f)

        # minimum number of steps in done state (-> 1 second)
        self._min_steps_terminated = int(1.0 * self._env.control_freq)
        # minimum number of steps in @ last waypoint before aborting the episode (-> 2 seconds)
        self._min_steps_last_wp = int(2.0 * self._env.control_freq)
        # timing and real time handler
        self._dt = 1.0 / self._env.control_freq
        self._rt_handler = RealTimeHandler(self._env.control_freq)

        self._ee_target_methods = self._create_ee_target_methods_dict()



    def _evaluate_controller_setup(self) -> None:
        """
        Evaluates the controller setup to get
        - controller input + output ranges
        - controller input type
        :return:
        """
        controller_dict = self.__get_controllers()
        self._get_action_mode(controller_dict)
        self._get_controller_ranges(controller_dict)

    def _get_action_mode(self, controllers_dict: dict[str, OperationalSpaceController]) -> None:
        """
        Check the action mode and store it in self._action_mode.
        It must be the same for all devices and can be either 'delta' or 'absolute'.

        :param controllers_dict: Dictionary with the controllers for the left and right arm
        """
        mode = None
        for controller in controllers_dict.values():
            if mode is None:
                mode = controller.input_type
            assert mode == controller.input_type, \
                f"[WP] Inconsistent action modes for the controllers.  {mode}, got {controller.input_type}"

        assert mode in ["delta", "absolute"], \
            f"[WP] Invalid action mode {mode}. Expected 'delta' or 'absolute' for all controllers."

        self._action_mode = mode

    def _get_controller_ranges(self, controllers_dict: dict[str, OperationalSpaceController]) -> None:
        """
        Get the input and output ranges of the controllers.

        :param controllers_dict: Dictionary with the controllers for the left and right arm
        """
        self._ctrl_input_min = {
            "left": controllers_dict["left"].input_min,
            "right": controllers_dict["right"].input_min
        }
        self._ctrl_input_max = {
            "left": controllers_dict["left"].input_max,
            "right": controllers_dict["right"].input_max
        }

        self._ctrl_output_min = {
            "left": controllers_dict["left"].output_min,
            "right": controllers_dict["right"].output_min
        }

        self._ctrl_output_max = {
            "left": controllers_dict["left"].output_max,
            "right": controllers_dict["right"].output_max
        }

    def __get_controllers(self) -> dict[str, OperationalSpaceController]:
        """
        Get the input and output ranges of the controllers.

        :return: Dictionary with the controllers for the left and right arm
        """
        controllers = {
            "left": None,
            "right": None
        }
        if len(self._env.robots) == 1:
            # single robot with two arms
            for name, part_controller in self._env.robots[0].composite_controller.part_controllers.items():
                if name not in ["left", "right"]:
                    continue
                assert type(part_controller) == OperationalSpaceController, \
                    f"[WP] Only OperationalSpaceController is supported, got {type(part_controller)}"
                controllers[name] = part_controller

        elif len(self._env.robots) == 2:
            mapped_names = ["right", "left"]
            # two separate robot arms
            for robot, mapped_name in zip(self._env.robots, mapped_names):
                for name, part_controller in robot.composite_controller.part_controllers.items():
                    if name not in ["right"]:
                        continue
                    assert type(part_controller) == OperationalSpaceController, \
                        f"[WP] Only OperationalSpaceController is supported, got {type(part_controller)}"

                    controllers[mapped_name] = part_controller

        else:
            raise ValueError(f"[WP] Invalid number of robots {len(self._env.robots)}. Expected 1 or 2.")

        return controllers

    def _create_waypoints(self, obs: OrderedDict = None) -> list[Waypoint]:
        """
        Create the list of waypoints.
        This method can be dependent on the environments initial state.
        :param obs: Observation after environment reset
        :return: List of waypoints
        """
        waypoints : list[Waypoint] = []

        for waypoint_dict in self._waypoint_cfg:
            waypoints.append(self.__create_waypoint(waypoint_dict, waypoints, obs))

        return waypoints

    def __create_waypoint(
            self, waypoint_dict: dict,
            previous_waypoints: list[Waypoint],
            obs: OrderedDict = None
    ) -> Waypoint:
        """
        Create a single waypoint.
        :param waypoint_dict: Dictionary containing the waypoint data
        :param previous_waypoints: List of previous waypoints
        :param obs: Observation after environment reset
        :return: Waypoint object
        """

        mapped_waypoint = {
            "id": waypoint_dict["id"],
            "description": waypoint_dict["description"],
            "min_duration": waypoint_dict.get("min_duration", DEFAULT_MIN_DURATION),
            "max_duration": waypoint_dict.get("max_duration", DEFAULT_MAX_DURATION),
            "must_reach": waypoint_dict.get("must_reach", DEFAULT_MUST_REACH),
            "targets": self.__create_targets(waypoint_dict, previous_waypoints, obs)
        }

        return Waypoint(mapped_waypoint)

    def __create_targets(
            self,
            waypoint_dict: dict,
            previous_waypoints: list[Waypoint],
            obs: OrderedDict = None
    ) -> list[dict]:
        """
        Create the list of targets for a waypoint.
        :param waypoint_dict:
        :param previous_waypoints:
        :param obs: Observation after environment reset
        :return:
        """

        mapped_targets = []
        for raw_target in waypoint_dict["targets"]:
            mapped_target = {
                "device": raw_target["device"],
                "pos_tol": raw_target.get("pos_tol", DEFAULT_POSITION_TOLERANCE),
                "rot_tol": raw_target.get("rot_tol", DEFAULT_ORIENTATION_TOLERANCE)
            }

            ####################################
            # 1 Check if ee_target is declared #
            ####################################
            ee_target = raw_target.get("ee_target", "")
            if ee_target:
                pattern = r"wp_(\d+)"
                if re.compile(pattern).match(ee_target):
                    previous_wp_id = int(ee_target.split("_")[1])
                    pos_quat_grip = self.__get_pos_quat_grip_from_previous_waypoint(raw_target["device"], previous_waypoints, previous_wp_id)
                else:
                    ee_target_method = self._ee_target_methods.get(ee_target, None)
                    if ee_target_method is None:
                        raise NotImplementedError(f"[WP] Method {ee_target} not implemented in {self.__class__.__name__}")

                    pos_quat_grip = ee_target_method(obs)

                mapped_target["pos"] = pos_quat_grip["pos"]
                mapped_target["quat"] = pos_quat_grip["quat"]
                mapped_target["grip"] = pos_quat_grip["grip"]

                mapped_targets.append(mapped_target)
                continue

            ####################################
            # 2 Map pos, quat, grip            #
            ####################################
            mapped_target["pos"] = self.__map_position(
                raw_target.get("pos", None),
                raw_target["device"],
                previous_waypoints,
            )
            mapped_target["quat"] = self.__map_orientation(
                raw_target.get("quat", None),
                raw_target.get("euler", None),
                raw_target.get("ax_angle", None),
                raw_target["device"],
                previous_waypoints,
            )
            mapped_target["grip"] = self.__map_gripper_target(raw_target.get("grip", None))

            mapped_targets.append(mapped_target)


        return mapped_targets

    @staticmethod
    def __get_pos_quat_grip_from_previous_waypoint(device: str, previous_waypoints: list[Waypoint], previous_wp_id: int) -> dict:
        """
        Get the position, orientation, and gripper state of a previous waypoint.
        :param device: Device name that the waypoint belongs to
        :param previous_waypoints: List of previous waypoints
        :param previous_wp_id: ID of the previous waypoint
        :return: Dictionary containing the position, orientation, and gripper state
        """
        # find the previous waypoint by id
        previous_wp = next(filter(lambda wp: wp.id == previous_wp_id, previous_waypoints), None)

        if previous_wp is None:
            raise ValueError(f"[WP] Previous waypoint with id {previous_wp_id} not found. Invalid yaml configuration.")

        previous_target = previous_wp.target.left if device == "robot_left" else previous_wp.target.right
        return {
            "pos": previous_target.xyz,
            "quat": previous_target.quat,
            "grip": previous_target.grip
        }

    def __map_position(self, pos: list | str | None, device: str, previous_waypoints: list[Waypoint]) -> np.array:
        """
        Map the position from a list to the target position.
        :param pos: Position as a list or string that needs to be mapped ("wp_<id>" or "wp_<id> + [dx, dy, dz]")
        :param device: device name that the position belongs to
        :param previous_waypoints: List of previous waypoints
        :return: Target position as a list
        """
        if type(pos) is list:
            if len(pos) != 3:
                raise ValueError(f"[WP] Invalid position list {pos}. Expected 3 values for [x, y, z]")
            return np.array(pos)

        if type(pos) is str:
            # "wp_<id>"
            pattern = r"^wp_(\d+)$"
            if re.compile(pattern).match(pos):
                previous_wp_id = int(pos.split("_")[1])
                pos_quat_grip = self.__get_pos_quat_grip_from_previous_waypoint(device, previous_waypoints, previous_wp_id)
                return pos_quat_grip["pos"]

            # "wp_<id> + [dx, dy, dz]"
            pattern = r"wp_(\d+)\s*\+\s*\[(-?\d*\.?\d+(?:,\s*-?\d*\.?\d+)*)\]"
            match = re.match(pattern, pos)
            if match:
                previous_wp_id = int(match.group(1))
                d_xyz = np.array([float(x) for x in match.group(2).split(",")])
                if len(d_xyz) != 3:
                    raise ValueError(f"[WP] Invalid position string {pos}. Expected 3 values for [dx, dy, dz]")
                previous_pos_quat_grip = self.__get_pos_quat_grip_from_previous_waypoint(device, previous_waypoints, previous_wp_id)
                return previous_pos_quat_grip["pos"] + d_xyz

            # invalid string
            raise ValueError(f"[WP] Invalid position string {pos}. Expected 'wp_<id>' or 'wp_<id> + [dx, dy, dz]'")

        raise ValueError(f"[WP] Invalid position. Either 'pos' must be a list or a string.")

    def __map_orientation(
            self,
            quat: list | str | None,
            euler: list | str | None,
            ax_angle: list | str | None,
            device: str, previous_waypoints: list[Waypoint]
    ) -> np.array:
        """
        Map the orientation from a list to the target orientation.
        :param quat: Orientation as a list [w, y, z, z]
            or string that needs to be mapped ("wp_<id>" or "wp_<id> * [x, y, z, w]" or "[x, y, z, w] * wp_<id>")
        :param euler: Orientation as a list [roll, pitch, yaw]
            or string that needs to be mapped ("wp_<id>" or "wp_<id> * [roll, pitch, yaw] or "[roll, pitch, yaw] * wp_<id>")
        :param ax_angle: Orientation as a list [vx, vy, vz, angle]
            or string that needs to be mapped ("wp_<id>" or "wp_<id> * [vx, vy, vz, angle] or "[vx, vy, vz, angle] * wp_<id>")
        :param device: device name that the orientation belongs to
        :param previous_waypoints: List of previous waypoints
        :return: Target quaternion as a list [x, y, z, w]
        """
        if quat is not None:
            return self.__map_orientation_fn(quat, device, previous_waypoints)
        elif euler is not None:
            return self.__map_orientation_fn(
                euler, device, previous_waypoints,
                num_values=3, map_fn=self.__euler_to_quat
            )
        elif ax_angle is not None:
            return self.__map_orientation_fn(
                ax_angle, device, previous_waypoints,
                map_fn=self.__ax_angle_to_quat
            )
        else:
            raise ValueError(f"[WP] Invalid orientation. Either 'quat', 'euler', or 'ax_angle' must be defined.")

    def __map_orientation_fn(
            self, rot: list | str,
            device: str,
            previous_waypoints: list[Waypoint],
            num_values: int = 4,
            map_fn: callable = lambda x: x
    ) -> np.array:
        """
        Map the quaternion from a list to the target quaternion.
        :param rot: Orientation as a list
            or string that needs to be mapped ("wp_<id>" or "wp_<id> * [...]" or "[...] * wp_<id>"),
            where [...] is a list of num_values floats
        :param device: device name that the orientation belongs to
        :param previous_waypoints: List of previous waypoints
        :param num_values: Number of values in the strings list
        :param map_fn: Function to map the values in the list to a quaternion
        :return: Target quaternion as a list [x, y, z, w]
        """
        if type(rot) is list:
            if len(rot) != num_values:
                raise ValueError(f"[WP] Invalid rotation list {rot}. Expected {num_values} values, got {len(rot)}")
            quat = map_fn(rot)
            return np.array(quat)

        if type(rot) is str:
            # "wp_<id>"
            pattern = r"^wp_(\d+)$"
            if re.compile(pattern).match(rot):
                previous_wp_id = int(rot.split("_")[1])
                pos_quat_grip = self.__get_pos_quat_grip_from_previous_waypoint(device, previous_waypoints, previous_wp_id)
                return pos_quat_grip["quat"]

            # "wp_<id> * [x, y, z, w]"
            pattern = r"wp_(\d+)\s*\*\s*\[(-?\d*\.?\d+(?:,\s*-?\d*\.?\d+)*)\]"
            match = re.match(pattern, rot)
            if match:
                previous_wp_id = int(match.group(1))
                d_rot= np.array([float(x) for x in match.group(2).split(",")])
                if len(d_rot) != num_values:
                    raise ValueError(f"[WP] Invalid rotation string {rot}. Expected {num_values} values, got {len(d_rot)}")
                d_quat = map_fn(d_rot)
                prev_quat = self.__get_pos_quat_grip_from_previous_waypoint(device, previous_waypoints, previous_wp_id)["quat"]
                return quat_multiply(prev_quat, d_quat)

            # "[x, y, z, w] * wp_<id>"
            pattern = r"\[(-?\d*\.?\d+(?:,\s*-?\d*\.?\d+)*)\]\s*\*\s*wp_(\d+)"
            match = re.match(pattern, rot)
            if match:
                d_rot = np.array([float(x) for x in match.group(1).split(",")])
                if len(d_rot) != num_values:
                    raise ValueError(f"[WP] Invalid rotation string {rot}. Expected {num_values} values, got {len(d_rot)}")
                previous_wp_id = int(match.group(2))
                d_quat = map_fn(d_rot)
                pre_quat = self.__get_pos_quat_grip_from_previous_waypoint(device, previous_waypoints, previous_wp_id)["quat"]
                return quat_multiply(d_quat, pre_quat)

            # invalid string
            raise ValueError(f"[WP] Invalid rotation string {rot}." + \
                             f" Expected 'wp_<id>' or 'wp_<id> * [x, y, z, w]' or '[x, y, z, w] * wp_<id>'" + \
                             f", got {rot}")

        raise ValueError(f"[WP] Invalid quaternion. Either 'quat' must be a list or a string.")

    @staticmethod
    def __euler_to_quat(euler: list) -> np.array:
        """
        Convert euler angles to a quaternion.
        :param euler: Euler angles as a list [roll, pitch, yaw]
        :return: Quaternion as array [x, y, z, w]
        """
        assert len(euler) == 3, f"[WP] Invalid euler angles {euler}. Expected 3 values for [roll, pitch, yaw]"

        return mat2quat(euler2mat(euler))

    @staticmethod
    def __ax_angle_to_quat(ax_angle: list) -> np.array:
        """
        Convert axis angle to a quaternion.
        :param ax_angle: Axis angle as a list [vx, vy, vz, angle]
        :return: Quaternion as array [x, y, z, w]
        """
        assert len(ax_angle) == 4, f"[WP] Invalid axis angle {ax_angle}. Expected 4 values for [vx, vy, vz, theta]"
        vx, vy, vz, angle = ax_angle
        aa_vec = np.array([vx, vy, vz])
        # robosuite expects the magnitude of the axis angle to be the rotation in radians
        aa_vec = (aa_vec / np.linalg.norm(aa_vec)) * angle
        return axisangle2quat(aa_vec)


    @staticmethod
    def __map_gripper_target(grip: str | None) -> float:
        """
        Map the gripper state from a string to the GripperState enum.
        :param grip: Gripper state as a string
        :return: Gripper state as a GripperState enum
        """
        if type(grip) is str and grip.casefold() == "OPEN".casefold():
            return GripperTarget.OPEN_VALUE
        elif type(grip) is str and grip.casefold() == "CLOSED".casefold():
            return GripperTarget.CLOSED_VALUE
        elif grip is None:
            return GripperTarget.OPEN_VALUE
        else:
            raise ValueError(f"[WP] Invalid gripper state {grip}. Valid values are 'OPEN', 'CLOSED'")

    def _get_action(self, current_obs: OrderedDict, waypoint: Waypoint) -> np.ndarray:
        """
        Get the action to reach the waypoint.
        The output is a clipped version of the Waypoint state and is scaled to the controller input range, e.g. [-1, 1].

        robosuite expects the following action components:
        - position action [x, y, z] or [dx, dy, dz]
        - orientation action as a 3D axis angle [vx, vy, vz] or [dvx, dvy, dvz],
            with the magnitude of the axis angle being the rotation in radians
        - gripper action [grip] (-1 => open, 1 => closed)

        - position action and orientation action are concatenated to the part action for the OSC
        - the gripper action is a separate action for the gripper controller

        :param current_obs: Current observation
        :param waypoint: Waypoint to reach
        :return: Action to reach the waypoint
        """
        current = TwoArmEEState.from_dict(current_obs, self._env.env_configuration)

        ctrl_input_min_lr = [self._ctrl_input_min["left"], self._ctrl_input_min["right"]]
        ctrl_input_max_lr = [self._ctrl_input_max["left"], self._ctrl_input_max["right"]]
        ctrl_output_min_lr = [self._ctrl_output_min["left"], self._ctrl_output_min["right"]]
        ctrl_output_max_lr = [self._ctrl_output_max["left"], self._ctrl_output_max["right"]]
        target_lr = [waypoint.target.left, waypoint.target.right]
        current_lr = [current.left, current.right]
        part_action_lr = []
        grip_action_lr = []

        for target, current, ctrl_input_min, ctrl_input_max, ctrl_output_min, ctrl_output_max, in zip(
                target_lr, current_lr, ctrl_input_min_lr, ctrl_input_max_lr, ctrl_output_min_lr, ctrl_output_max_lr
        ):
            # Clip the translation
            pos_target = target.xyz
            pos_current = current.xyz
            pos_delta = clip_translation(pos_target - pos_current, ctrl_output_min[:3], ctrl_output_max[:3])
            if self._action_mode == "absolute":
                pos_action = pos_current + pos_delta
            elif self._action_mode == "delta":
                pos_action = pos_delta
            else:
                raise ValueError(f"[WP] Invalid action mode {self._action_mode}. Expected 'delta' or 'absolute'")

            quat_target = target.quat
            quat_current = current.quat
            quat_delta = clip_quat_by_axisangle( # output quat of type wxyz
                quat_multiply(quat_target, quat_inverse(quat_current)),
                ctrl_output_min[3:],
                ctrl_output_max[3:]
            )
            if self._action_mode == "absolute":
                rot_target = quat_multiply(quat_delta, quat_current)
                rot_action = quat2axisangle(rot_target)
            else:
                rot_action = quat2axisangle(quat_delta)

            part_action = np.concatenate([pos_action, rot_action])

            # Rescale the input range
            if self._action_mode == "delta":
                # (this is the reverse of
                # https://github.com/ARISE-Initiative/robosuite/blob/0926cbec81bf19ff7667d387b55da8b8714647ea/robosuite/controllers/parts/controller.py#L149)
                action_scale = abs(ctrl_output_max - ctrl_output_min) / abs(ctrl_input_max - ctrl_input_min)
                action_output_transform = (ctrl_output_max + ctrl_output_min) / 2.0
                action_input_transform = (ctrl_input_max + ctrl_input_min) / 2.0

                part_action = (part_action - action_output_transform) / action_scale + action_input_transform

            part_action_lr.append(part_action)

            # Set GripperState
            grip_action = np.array([target.grip])
            grip_action_lr.append(grip_action)

        action = self.__compose_action(part_action_lr, grip_action_lr)

        return action

    def __compose_action(self, part_action_lr: list[np.ndarray], grip_action_lr: list[np.ndarray]) -> np.ndarray:
        """
        Compose the action for the two arms.

        robosuite order: right before left

        :param part_action_lr: the action controlling the position and orientation of the end-effector
        :param grip_action_lr: the action controlling the gripper state
        :return: total action
        """
        dof = self._env.action_spec[0].shape[0]
        if dof == 12: # no gripper
            return np.concatenate(part_action_lr[::-1])

        if dof == 14: # gripper + single arms
            if self._env.env_configuration == "single-robot":
                # first the part actions, then the gripper actions
                return np.concatenate(part_action_lr[::-1] + grip_action_lr[::-1])
            else:
                # part + gripper + part + gripper
                return np.concatenate([
                    part_action_lr[1], grip_action_lr[1], part_action_lr[0], grip_action_lr[0]
                ])

        raise NotImplementedError(f"[WP] Unsupported action space with {dof} DoF.")

    def _run_episode(
        self,
        render : bool = False,
        target_real_time : bool = False,
        randomize : bool = False
    ) -> bool:
        """
        Run the expert agent in the environment.
        :param render: Whether to render the environment
        :param target_real_time: Whether to render in real time
        :param randomize: Whether to randomize the environment

        :return: True if the episode was successful
        """
        if target_real_time and not (render and self._env.has_renderer):
            print("[TwoArmWaypointExpertBase - INFO] Ignoring real time rendering as the environment " + \
                  "does not have a renderer and/or render==False.")
            target_real_time = False

        # reset the environment
        self._env.deterministic_reset = not randomize
        obs = self._env.reset()

        waypoints = self._create_waypoints(obs)

        steps_terminated = 0
        steps_in_last_wp = 0
        success = False
        unreachable = False
        for n, waypoint in enumerate(waypoints):
            is_last_wp = n == len(waypoints) - 1
            wp_step = 0
            self._rt_handler.reset()
            while True:
                action = self._get_action(obs, waypoint)
                obs, _, _, _ = self._env.step(action)
                done = self._env._check_success()
                if done:
                    steps_terminated += 1
                    if steps_terminated >= self._min_steps_terminated:
                        success = True
                        break
                else:
                    steps_terminated = 0

                if render:
                    self._env.render()
                wp_step += 1
                reached, unreachable = waypoint.is_reached_by(
                    obs, self._env.env_configuration, wp_step * self._dt
                )

                if reached and not is_last_wp:
                    break
                elif reached and is_last_wp:
                    steps_in_last_wp += 1
                    if steps_in_last_wp >= self._min_steps_last_wp:
                        success = done
                        break
                elif unreachable:
                    break

                if target_real_time:
                    self._rt_handler.sleep()

            if success:
                break
            if unreachable:
                print(f"[INFO] Waypoint {waypoint.id} ({waypoint.description}) could not be reached. Aborting this episode.")
                break



        if success:
            print(f"[INFO] Episode finished successful.")
        else:
            print(f"[INFO] Episode finished unsuccessful.")

        if self._env.viewer is not None:
            self._env.viewer.close()

        return success

    def visualize(
        self,
        num_episodes : int = 1,
        visualize_targets : bool = False
    ) -> None:
        """
        Visualize the expert agent in the environment.
        :param num_episodes: Number of episodes to visualize
        :param visualize_targets: Whether to visualize the target positions
        """
        if self.visualize:
            self._env = TargetVisualizationWrapper(self._env)

        for _ in range(num_episodes):
            _ = self._run_episode(
                render=True,
                target_real_time=True,
                randomize=True
            )

        if visualize_targets:
            self._env = self._env.unwrapped()

    def collect_data(self,
        out_dir : str,
        num_successes : int,
        env_config: dict,
        render: bool = False,
        target_real_time: bool = False
    ):
        """
        Collect demonstration from the expert agent in the environment.

        :param out_dir: Output directory for the data
        :param num_successes: Number of successful episodes to collect
        :param env_config: Configuration of the environment, needed to ensure reproducibility
        :param render: Whether to render the environment
        :param target_real_time: Whether to render in real time
        """
        tmp_dir = os.path.join("/tmp", "bil_wp", str(time.time()).replace(".", "_"))

        self._env = DataCollectionWrapper(
            self._env,
            directory=tmp_dir,
        )

        success_count = 0
        count = 0
        while success_count < num_successes:
            success_count += self._run_episode(
                render=render,
                target_real_time=target_real_time,
                randomize=True
            )
            count += 1
            print(f"[INFO] Collected {success_count}/{num_successes} successful demonstrations after {count} episodes.")

        self._env.close()

        env_info = json.dumps(env_config)
        gather_demonstrations_as_hdf5(tmp_dir, out_dir, env_info)

        shutil.rmtree(tmp_dir)


    def _create_ee_target_methods_dict(self) -> dict:
        """
        Create a dictionary of methods that return the position, orientation, and gripper state of a device.

        All methods take the observation after reset as input.
        :return: Dictionary of methods
        """
        return {
            "initial_state_left": self._initial_state_left,
            "initial_state_right": self._initial_state_right,
        }

    def _initial_state_left(self, obs: OrderedDict = None) -> dict:
        """
        Initial state for the left arm.

        :param obs: observation after reset
        :return: dictionary with the target position, orientation, and gripper state
        """
        state: EEState = TwoArmEEState.from_dict(obs, env_config=self._env.env_configuration).left
        dct = {
            "pos": state.xyz,
            "quat": state.quat,
            "grip": GripperTarget.OPEN_VALUE
        }
        return dct

    def _initial_state_right(self, obs: OrderedDict = None) -> dict:
        """
        Initial state for the right arm.

        :param obs: observation after reset
        :return: dictionary with the target position, orientation, and gripper state
        """
        state: EEState = TwoArmEEState.from_dict(obs, env_config=self._env.env_configuration).right
        dtc = {
            "pos": state.xyz,
            "quat": state.quat,
            "grip": GripperTarget.OPEN_VALUE
        }
        return dtc




