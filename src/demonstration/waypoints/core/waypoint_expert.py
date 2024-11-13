import os.path
from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np
import yaml
import re

from robosuite.controllers.parts.arm import OperationalSpaceController
from robosuite.environments.manipulation.two_arm_env import TwoArmEnv
from transforms3d.euler import euler2quat, mat2euler, quat2euler
from transforms3d.quaternions import qmult, qinverse, axangle2quat

from src.demonstration.waypoints.core.waypoint import Waypoint, DEFAULT_MUST_REACH, DEFAULT_MIN_DURATION, \
    DEFAULT_MAX_DURATION, DEFAULT_POSITION_TOLERANCE, DEFAULT_ORIENTATION_TOLERANCE
from src.utils.clipping import clip_translation, clip_quat
from src.utils.constants import MAX_DELTA_TRANSLATION, MAX_DELTA_ROTATION
from src.utils.paths import WAYPOINTS_DIR
from src.utils.real_time import RealTimeHandler
from src.utils.robot_states import TwoArmEEState
from src.utils.robot_targets import GripperTarget


class TwoArmWaypointExpertBase(ABC):
    """
    Class for an expert agent that acts in the environment
    by following a predefined trajectory of waypoints.
    """
    def __init__(
        self,
        environment : TwoArmEnv,
        waypoints_file : str,
        max_delta_translation : float = MAX_DELTA_TRANSLATION,
        max_delta_rotation : float = MAX_DELTA_ROTATION,
    ) -> None:
        """
        Constructor for the WaypointExpert class.
        :param environment: Environment in which the expert agent acts
        :param waypoints_file: File containing the waypoints n $WAYPOINTS_DIR
        :param max_delta_translation: Maximum translation distance between current position and action output
        :param max_delta_rotation: Maximum rotation angle between current orientation and action output
        """
        full_waypoints_path = os.path.join(WAYPOINTS_DIR, waypoints_file)
        assert os.path.isfile(full_waypoints_path), f"Waypoints file {full_waypoints_path} not found"
        self._env = environment

        self._max_delta_translation = max_delta_translation
        self._max_delta_rotation = max_delta_rotation

        with open(full_waypoints_path, 'r') as f:
            self._waypoint_cfg = yaml.safe_load(f)

        self._action_mode = self._check_action_mode()

        # minimum number of steps in done state (-> 1 second)
        self._min_steps_terminated = int(1.0 * self._env.control_freq)
        self._dt = 1.0 / self._env.control_freq
        self._rt_handler = RealTimeHandler(self._env.control_freq)

        self._ee_target_methods = self._create_ee_target_methods_dict()

    def _check_action_mode(self) -> str:
        """
        Check the action mode.
        It must be the same for all devices and can be either 'delta' or 'absolute'.
        :return: The action mode
        """
        mode = None
        for robot in self._env.robots:
            for name, part_controller in robot.composite_controller.part_controllers.items():
                if name not in ["left", "right"]:
                    continue
                if type(part_controller) != OperationalSpaceController:
                    raise ValueError(f"[WP] Only OperationalSpaceController is supported, got {type(part_controller)}")
                if mode is None:
                    mode = part_controller.input_type
                elif mode != part_controller.input_type:
                    raise ValueError(
                        f"[WP] Inconsistent action modes for the devices.  {mode}, got {part_controller.input_type}")

        assert mode in ["delta", "absolute"], \
            f"[WP] Invalid action mode {mode}. Expected 'delta' or 'absolute' for all arms."

        return mode

    def _create_waypoints(self) -> list[Waypoint]:
        """
        Create the list of waypoints.
        This method can be dependent on the environments initial state.
        :return: List of waypoints
        """
        waypoints : list[Waypoint] = []

        for waypoint_dict in self._waypoint_cfg:
            waypoints.append(self.__create_waypoint(waypoint_dict, waypoints))

        return waypoints

    def __create_waypoint(self, waypoint_dict: dict, previous_waypoints: list[Waypoint]) -> Waypoint:
        """
        Create a single waypoint.
        :param waypoint_dict: Dictionary containing the waypoint data
        :param previous_waypoints: List of previous waypoints
        :return: Waypoint object
        """

        mapped_waypoint = {
            "id": waypoint_dict["id"],
            "description": waypoint_dict["description"],
            "min_duration": waypoint_dict.get("min_duration", DEFAULT_MIN_DURATION),
            "max_duration": waypoint_dict.get("max_duration", DEFAULT_MAX_DURATION),
            "must_reach": waypoint_dict.get("must_reach", DEFAULT_MUST_REACH),
            "targets": self.__create_targets(waypoint_dict, previous_waypoints)
        }

        return Waypoint(mapped_waypoint)

    def __create_targets(self, waypoint_dict: dict, previous_waypoints: list[Waypoint]) -> list[dict]:
        """
        Create the list of targets for a waypoint.
        :param waypoint_dict:
        :param previous_waypoints:
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

                    pos_quat_grip = ee_target_method()

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
            or string that needs to be mapped ("wp_<id>" or "wp_<id> * [w, x, y, z]" or "[w, x, y, z] * wp_<id>")
        :param euler: Orientation as a list [roll, pitch, yaw]
            or string that needs to be mapped ("wp_<id>" or "wp_<id> * [roll, pitch, yaw] or "[roll, pitch, yaw] * wp_<id>")
        :param ax_angle: Orientation as a list [vx, vy, vz, angle]
            or string that needs to be mapped ("wp_<id>" or "wp_<id> * [vx, vy, vz, angle] or "[vx, vy, vz, angle] * wp_<id>")
        :param device: device name that the orientation belongs to
        :param previous_waypoints: List of previous waypoints
        :return: Target quaternion as a list [w, x, y, z]
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
        :return: Target quaternion as a list [w, x, y, z]
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

            # "wp_<id> * [w, x, y, z]"
            pattern = r"wp_(\d+)\s*\*\s*\[(-?\d*\.?\d+(?:,\s*-?\d*\.?\d+)*)\]"
            match = re.match(pattern, rot)
            if match:
                previous_wp_id = int(match.group(1))
                d_rot= np.array([float(x) for x in match.group(2).split(",")])
                if len(d_rot) != num_values:
                    raise ValueError(f"[WP] Invalid rotation string {rot}. Expected {num_values} values, got {len(d_rot)}")
                d_quat = map_fn(d_rot)
                prev_quat = self.__get_pos_quat_grip_from_previous_waypoint(device, previous_waypoints, previous_wp_id)["quat"]
                return qmult(prev_quat, d_quat)

            # "[w, x, y, z] * wp_<id>"
            pattern = r"\[(-?\d*\.?\d+(?:,\s*-?\d*\.?\d+)*)\]\s*\*\s*wp_(\d+)"
            match = re.match(pattern, rot)
            if match:
                d_rot = np.array([float(x) for x in match.group(1).split(",")])
                if len(d_rot) != num_values:
                    raise ValueError(f"[WP] Invalid rotation string {rot}. Expected {num_values} values, got {len(d_rot)}")
                previous_wp_id = int(match.group(2))
                d_quat = map_fn(d_rot)
                pre_quat = self.__get_pos_quat_grip_from_previous_waypoint(device, previous_waypoints, previous_wp_id)["quat"]
                return qmult(d_quat, pre_quat)

            # invalid string
            raise ValueError(f"[WP] Invalid rotation string {rot}." + \
                             f" Expected 'wp_<id>' or 'wp_<id> * [w, x, y, z]' or '[w, x, y, z] * wp_<id>'" + \
                             f", got {rot}")

        raise ValueError(f"[WP] Invalid quaternion. Either 'quat' must be a list or a string.")

    @staticmethod
    def __euler_to_quat(euler: list) -> np.array:
        """
        Convert euler angles to a quaternion.
        :param euler: Euler angles as a list [roll, pitch, yaw]
        :return: Quaternion as array [w, x, y, z]
        """
        assert len(euler) == 3, f"[WP] Invalid euler angles {euler}. Expected 3 values for [roll, pitch, yaw]"
        return euler2quat(*euler)

    @staticmethod
    def __ax_angle_to_quat(ax_angle: list) -> np.array:
        """
        Convert axis angle to a quaternion.
        :param ax_angle: Axis angle as a list [vx, vy, vz, angle]
        :return: Quaternion as array [w, x, y, z]
        """
        assert len(ax_angle) == 4, f"[WP] Invalid axis angle {ax_angle}. Expected 4 values for [vx, vy, vz, theta]"
        vx, vy, vz, angle = ax_angle
        return axangle2quat(
            vector=[vx, vy, vz],
            theta=angle,
            is_normalized=False
        )

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

    def _get_action(self, current_state: OrderedDict, waypoint: Waypoint) -> np.ndarray:
        """
        Get the action to reach the waypoint.
        The output is a clipped version of the Waypoint state to
        respect self._max_delta_translation and self._max_delta_rotation.
        :param current_state: Current state of the device
        :param waypoint: Waypoint to reach
        :return: Action to reach the waypoint
        """
        max_delta_translation = self._max_delta_translation
        max_delta_rotation = self._max_delta_rotation

        current = TwoArmEEState.from_dict(current_state, self._env.env_configuration)

        target_lr = [waypoint.target.left, waypoint.target.right]
        current_lr = [current.left, current.right]
        action_lr = []

        for target, current in zip(target_lr, current_lr):
            # Clip the translation
            pos_target = target.xyz
            pos_current = current.xyz
            pos_delta = clip_translation(pos_target - pos_current, max_delta_translation)
            if self._action_mode == "absolute":
                pos_action = pos_current + pos_delta
            elif self._action_mode == "delta":
                pos_action = pos_delta
                # TODO: rescale from range [-0.05, 0.05] to range [-1, 1], better: depending on the controller config
                # => see https://github.com/ARISE-Initiative/robosuite/blob/0926cbec81bf19ff7667d387b55da8b8714647ea/robosuite/controllers/parts/controller.py#L149

            else:
                raise ValueError(f"[WP] Invalid action mode {self._action_mode}. Expected 'delta' or 'absolute'")

            # Clip the rotation
            quat_target = target.quat
            quat_current = current.quat
            quat_delta = clip_quat(
                qmult(quat_target, qinverse(quat_current)),
                max_delta_rotation
            )
            if self._action_mode == "absolute":
                rot_target = qmult(quat_delta, quat_current)
                rot_action = quat2euler(rot_target)
            else:
                rot_action = quat2euler(quat_delta)

                # TODO: rescale from range [-0.5, 0.5] to range [-1, 1], better: depending on the controller config
                # => see https://github.com/ARISE-Initiative/robosuite/blob/0926cbec81bf19ff7667d387b55da8b8714647ea/robosuite/controllers/parts/controller.py#L149

            # Set GripperState
            grip_action = np.array([target.grip])

            action_lr.append(
                np.concatenate([pos_action, rot_action, grip_action])
            )

        action = np.concatenate(action_lr)
        return action

    def _run_episode(
        self,
        render : bool = False,
        target_real_time : bool = False,
    ) -> bool:
        """
        Run the expert agent in the environment.
        :param render: Whether to render the environment
        :param target_real_time: Whether to render in real time

        :return: True if the episode was successful
        """
        if target_real_time and not (render and self._env.has_renderer):
            print("[TwoArmWaypointExpertBase - INFO] Ignoring real time rendering as the environment " + \
                  "does not have a renderer and/or render==False.")
            target_real_time = False

        # reset the environment
        obs = self._env.reset()

        waypoints = self._create_waypoints()

        steps_terminated = 0
        success = False
        for n, waypoint in enumerate(waypoints):
            is_last_wp = n == len(waypoints) - 1
            wp_step = 0
            self._rt_handler.reset()
            while True:
                action = self._get_action(obs, waypoint)
                obs, _, done, _ = self._env.step(action)
                if render:
                    self._env.render()
                wp_step += 1
                reached, unreachable = waypoint.is_reached_by(
                    obs, self._env.env_configuration, wp_step * self._dt
                )

                if reached and not is_last_wp:
                    break
                elif unreachable:
                    break

                if done:
                    steps_terminated += 1
                    if steps_terminated >= self._min_steps_terminated:
                        success = True
                        break
                else:
                    steps_terminated = 0

                if target_real_time:
                    self._rt_handler.sleep()

            if success:
                print(f"[INFO] Episode finished successful.")

            if unreachable:
                print(f"[INFO] Waypoint {waypoint.id} ({waypoint.description}) could not be reached. Aborting this episode.")
                break

        return success

    def dispose(self) -> None:
        """
        Dispose the expert agent.
        """
        self._env.close()

    def visualize(
        self,
        num_episodes : int = 1
    ) -> None:
        """
        Visualize the expert agent in the environment.
        :param num_episodes: Number of episodes to visualize
        """
        for _ in range(num_episodes):
            _ = self._run_episode(render=True)

    def collect_data(self,
        out_dir : str,
        num_successes : int,
        render: bool = False,
        target_real_time: bool = False
    ) -> None:
        """
        Collect demonstration from the expert agent in the environment.

        :param out_dir: Output directory for the data
        :param num_successes: Number of successful episodes to collect
        :param render: Whether to render the environment
        :param target_real_time: Whether to render in real time
        """
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def _create_ee_target_methods_dict(self) -> dict:
        """
        Create a dictionary of methods that return the position, orientation, and gripper state of a device.
        :return: Dictionary of methods
        """
        return {}




