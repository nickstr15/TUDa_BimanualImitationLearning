import os.path
import time
from abc import ABC, abstractmethod

import numpy as np
import yaml
import re

from transforms3d.quaternions import qmult, qinverse

from src.control.utils.ee_state import EEState
from src.control.utils.enums import GripperState
from src.demonstration.data_collection.data_collection_wrapper import DataCollectionWrapper
from src.demonstration.data_collection.hdf5 import gather_demonstrations_as_hdf5
from src.demonstration.waypoints.core.waypoint import Waypoint
from src.environments.core.action import OSAction
from src.environments.core.enums import ActionMode
from src.environments.core.environment_interface import IEnvironment
from src.utils.clipping import clip_translation, clip_quat
from src.utils.constants import MAX_DELTA_TRANSLATION, MAX_DELTA_ROTATION
from src.utils.paths import WAYPOINTS_DIR
from src.utils.real_time import RealTimeHandler

class WaypointExpertBase(ABC):
    """
    Class for an expert agent that acts in the environment
    by following a predefined trajectory of waypoints.
    """
    def __init__(
        self,
        environment : IEnvironment,
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

        self._action_mode = self._env.action_mode

        # minimum number of steps in done state (-> 1 second)
        self._min_steps_terminated = int(1.0 * self._env.render_fps)
        self._dt = 1.0 / self._env.render_fps
        self._rt_handler = RealTimeHandler(self._env.render_fps)

    def _create_waypoints(self) -> list[Waypoint]:
        """
        Create the list of waypoints.
        This method can be dependent on the environments initial state.
        :return: List of waypoints
        """
        waypoints : list[Waypoint] = []

        for waypoint_dict in self._waypoint_cfg["waypoints"]:
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
            "min_duration": waypoint_dict.get("min_duration", 0.0),
            "max_duration": waypoint_dict.get("max_duration", 100.0),
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
                "position_tolerance": raw_target.get("position_tolerance", 0.001),
                "orientation_tolerance": raw_target.get("orientation_tolerance", 0.001)
            }

            ####################################
            # 1 Check if ee_target is declared #
            ####################################
            ee_target = raw_target.get("ee_target", "")
            if ee_target:
                if re.compile("^wp_\d+$").match(ee_target):
                    previous_wp_id = int(ee_target.split("_")[1])
                    pos_quat_grip = self.__get_pos_quat_grip_from_previous_waypoint(raw_target["device"], previous_waypoints, previous_wp_id)
                else:
                    if not hasattr(self, f"__{ee_target}"):
                        raise ValueError(f"[WP] Invalid ee_target {ee_target}. Method __{ee_target} not found in {self.__class__.__name__}")
                    ee_target_method = getattr(self, f"__{ee_target}")
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
                raw_target.get("axis_angle", None),
                raw_target["device"],
                previous_waypoints,
            )
            mapped_target["grip"] = self.__map_gripper_state(raw_target.get("grip", None))

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

        previous_target = previous_wp.targets[device]
        return {
            "pos": previous_target.get_xyz(),
            "quat": previous_target.get_quat(),
            "grip": previous_target.get_gripper_state()
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
            return np.array(pos)
        elif type(pos) is str:
            if re.compile("^wp_\d+$").match(pos):
                previous_wp_id = int(pos.split("_")[1])
                pos_quat_grip = self.__get_pos_quat_grip_from_previous_waypoint(device, previous_waypoints, previous_wp_id)
                return pos_quat_grip["pos"]
            elif #TODO
                previous_wp_id = int(pos.split("_")[1])
                previous_pos_quat_grip = self.__get_pos_quat_grip_from_previous_waypoint(device, previous_waypoints, previous_wp_id)
                dx, dy, dz = ... #TODO
                return previous_pos_quat_grip["pos"] + np.array([dx, dy, dz])
            else:
                raise ValueError(f"[WP] Invalid position string {pos}. Expected 'wp_<id>' or 'wp_<id> + [dx, dy, dz]'")
        else:
            raise ValueError(f"[WP] Invalid position. Either 'pos' must be a list or a string.")

    def __map_orientation(
            self,
            quat: list | str | None,
            euler: list | str | None,
            axis_angle: list | str | None,
            device: str, previous_waypoints: list[Waypoint]
    ) -> list:
        """
        Map the orientation from a list to the target orientation.
        :param quat: Orientation as a list [w, y, z, z]
            or string that needs to be mapped ("wp_<id>" or "wp_<id> * [w, x, y, z]" or "[w, x, y, z] * wp_<id>")
        :param euler: Orientation as a list [roll, pitch, yaw]
            or string that needs to be mapped ("wp_<id>" or "wp_<id> * [roll, pitch, yaw] or "[roll, pitch, yaw] * wp_<id>")
        :param axis_angle: Orientation as a list [vx, vy, vz, angle]
            or string that needs to be mapped ("wp_<id>" or "wp_<id> * [vx, vy, vz, angle] or "[vx, vy, vz, angle] * wp_<id>")
        :param device: device name that the orientation belongs to
        :param previous_waypoints: List of previous waypoints
        :return: Target quaternion as a list [w, x, y, z]
        """
        if quat is not None:
            return self.__map_quat(quat, device, previous_waypoints)
        elif euler is not None:
            return self.__map_euler(euler, device, previous_waypoints)
        elif axis_angle is not None:
            return self.__map_axis_angle(axis_angle, device, previous_waypoints)
        else:
            raise ValueError(f"[WP] Invalid orientation. Either 'quat', 'euler', or 'axis_angle' must be defined.")

    @staticmethod
    def __map_quat(quat: list | str, device: str, previous_waypoints: list[Waypoint]) -> list:
        """
        Map the quaternion from a list to the target quaternion.
        :param quat: Orientation as a list [w, y, z, z]
            or string that needs to be mapped ("wp_<id>" or "wp_<id> * [w, x, y, z]" or "[w, x, y, z] * wp_<id>")
        :param device: device name that the orientation belongs to
        :param previous_waypoints: List of previous waypoints
        :return: Target quaternion as a list [w, x, y, z]
        """
        # TODO implement this
        raise NotImplementedError

    @staticmethod
    def __map_euler(euler: list | str, device: str, previous_waypoints: list[Waypoint]) -> list:
        """
        Map the euler angles from a list to the target quaternion.
        :param euler: Orientation as a list [roll, pitch, yaw]
            or string that needs to be mapped ("wp_<id>" or "wp_<id> * [roll, pitch, yaw] or "[roll, pitch, yaw] * wp_<id>")
        :param device: device name that the orientation belongs to
        :param previous_waypoints: List of previous waypoints
        :return: Target quaternion as a list [w, x, y, z]
        """
        # TODO implement this
        raise NotImplementedError

    @staticmethod
    def __map_axis_angle(axis_angle: list | str, device: str, previous_waypoints: list[Waypoint]) -> list:
        """
        Map the axis angle from a list to the target quaternion.
        :param axis_angle: Orientation as a list [vx, vy, vz, angle]
            or string that needs to be mapped ("wp_<id>" or "wp_<id> * [vx, vy, vz, angle] or "[vx, vy, vz, angle] * wp_<id>")
        :param device: device name that the orientation belongs to
        :param previous_waypoints: List of previous waypoints
        :return: Target quaternion as a list [w, x, y, z]
        """
        # TODO implement this
        raise NotImplementedError

    @staticmethod
    def __map_gripper_state(grip: str | int) -> GripperState:
        """
        Map the gripper state from a string to the GripperState enum.
        :param grip: Gripper state as a string
        :return: Gripper state as a GripperState enum
        """
        if type(grip) is str and grip.casefold() == "OPEN".casefold():
            return GripperState.OPEN
        elif type(grip) is str and grip.casefold() == "CLOSED".casefold():
            return GripperState.CLOSED
        elif type(grip) is int and grip == 255:
            return GripperState.OPEN
        elif type(grip) is int and grip == 0:
            return GripperState.CLOSED
        else:
            raise ValueError(f"[WP] Invalid gripper state {grip}. Valid values are 'OPEN', 255, 'CLOSED', 0")

    def _get_action(self, current_state: dict, waypoint: Waypoint) -> OSAction:
        """
        Get the action to reach the waypoint.
        The output is a clipped version of the Waypoint state to
        respect self._max_delta_translation and self._max_delta_rotation.
        :param current_state: Current state of the device
        :param waypoint: Waypoint to reach
        :return: Action to reach the waypoint
        """
        assert current_state.keys() == waypoint.targets.keys(), "Current state and waypoint targets do not match"

        targets = {
            name : EEState()
            for name in waypoint.targets.keys()
        }

        max_delta_translation = self._max_delta_translation
        max_delta_rotation = self._max_delta_rotation

        for name, target in waypoint.targets.items():
            current = current_state[name]

            # Clip the translation
            pos_target = target.get_xyz()
            pos_current = current.get_xyz()
            pos_delta = clip_translation(pos_target - pos_current, max_delta_translation)
            if self._action_mode == ActionMode.ABSOLUTE:
                targets[name].set_xyz(pos_delta + pos_current)
            elif self._action_mode == ActionMode.RELATIVE:
                targets[name].set_xyz(pos_delta)

            # Clip the rotation
            quat_target = target.get_quat()
            quat_current = current.get_quat()
            quat_delta = clip_quat(
                qmult(quat_target, qinverse(quat_current)),
                max_delta_rotation
            )
            if self._action_mode == ActionMode.ABSOLUTE:
                targets[name].set_quat(qmult(quat_delta, quat_current))
            else:
                targets[name].set_quat(quat_delta)

            # Set GripperState
            targets[name].set_gripper_state(target.get_gripper_state())

        return OSAction(targets)

    def _run_episode(
        self,
        render : bool = False,
        target_real_time : bool = False,
        random : bool = True,
    ) -> bool:
        """
        Run the expert agent in the environment.
        :param render: Whether to render the environment
        :param target_real_time: Whether to render in real time
        :param random: Whether to randomize the initial configuration of each episode

        :return: True if the episode was successful
        """
        if target_real_time and not (render and self._env.render_mode == "human"):
            print("[INFO] Real time rendering requires rendering in human mode, ignoring real time rendering.")
            target_real_time = False

        # reset the environment
        reset_options = dict(
            randomize=random,
        )
        self._env.reset(options=reset_options)
        current_ee_states = self._env.get_robot_ee_states()

        waypoints = self._create_waypoints()

        steps_terminated = 0
        success = False
        for waypoint in waypoints:
            wp_step = 0
            reached, unreachable = False, False
            self._rt_handler.reset()
            while (not reached) and (not unreachable):
                action = self._get_action(current_ee_states, waypoint)
                observation, _, terminated, _, _ = self._env.step(action)
                if render:
                    self._env.render()
                wp_step += 1
                current_ee_states = self._env.get_robot_ee_states()
                reached, unreachable = waypoint.is_reached_by(
                    current_ee_states, wp_step * self._dt
                )

                if unreachable:
                    break


                if terminated:
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
        tmp_directory = "/tmp/bil_wp/{}".format(str(time.time()).replace(".", "_"))
        self._env = DataCollectionWrapper(self._env, tmp_directory)

        success_count = 0
        while success_count < num_successes:
            success_count += self._run_episode(render=render, target_real_time=target_real_time)

        self._env.close()

        gather_demonstrations_as_hdf5(tmp_directory, out_dir, self._env.args)

        self._env.clean_up()




