import numpy as np
from collections import OrderedDict
import re

from numpy.ma.core import max_filler
from robosuite.utils.transform_utils import quat_multiply, mat2quat, euler2mat, axisangle2quat

from src.utils.robot_states import TwoArmEEState
from src.utils.robot_targets import EETarget, TwoArmEETarget, GripperTarget

DEFAULT_POSITION_TOLERANCE = 0.01
DEFAULT_ORIENTATION_TOLERANCE = np.deg2rad(5)
DEFAULT_MIN_DURATION = 0.5 # before this time, the waypoint is not considered reached
DEFAULT_MAX_DURATION = 10.0 # after this time, the waypoint is considered as unreachable
DEFAULT_MUST_REACH = True # if True, the waypoint must be reached before DEFAULT_MAX_DURATION,
                          # if False the episode can continue, even if the waypoint is not reached
DEFAULT_USES_FEEDBACK = False # if True, the waypoint is recomputed every step, based on the current observation

class Waypoint:
    """
    A class to represent a waypoint.
    """
    def __init__(
        self,
        waypoint_dict: dict,
        known_waypoints: list,
        obs: OrderedDict,
        waypoint_expert
    ) -> None:
        """
        Constructor for the Waypoint class.
        :param waypoint_dict: Dictionary containing the waypoint data
        :param known_waypoints: List of known/already computed waypoints
        :param obs: Observation after environment reset
        :param waypoint_expert: Waypoint expert to compute the special targets
        """
        self._waypoint_expert = waypoint_expert

        self._waypoint_dict = waypoint_dict
        self._load_waypoint(known_waypoints, obs)

    def _load_waypoint(
            self,
            known_waypoints: list,
            obs: OrderedDict
    ) -> None:
        """
        Load the waypoint data from the dictionary.
        :param known_waypoints: List of known/already computed waypoints
        :param obs: Observation after environment reset
        :return:
        """
        self._id = self._waypoint_dict["id"]
        self._des = self._waypoint_dict["description"]
        self._uses_feedback = self._waypoint_dict.get("uses_feedback", DEFAULT_USES_FEEDBACK)
        self._min_duration = self._waypoint_dict.get("min_duration", DEFAULT_MIN_DURATION)
        self._max_duration = self._waypoint_dict.get("max_duration", DEFAULT_MAX_DURATION)
        self._must_reach = self._waypoint_dict.get("must_reach", DEFAULT_MUST_REACH)

        raw_targets = self.__create_targets(self._waypoint_dict, known_waypoints, obs)

        left = EETarget()
        right = EETarget()

        for target in raw_targets:
            if target["device"] == "robot_left":
                left = EETarget(
                    xyz=target["pos"],
                    quat=target["quat"],
                    grip=target["grip"],
                    pos_tol=target.get("pos_tol", DEFAULT_POSITION_TOLERANCE),
                    ori_tol=target.get("rot_tol", DEFAULT_ORIENTATION_TOLERANCE),
                    max_vel_pos=target.get("max_vel_pos", None),
                    max_vel_ori=target.get("max_vel_ori", None)
                )
            elif target["device"] == "robot_right":
                right = EETarget(
                    xyz=target["pos"],
                    quat=target["quat"],
                    grip=target["grip"],
                    pos_tol=target.get("pos_tol", DEFAULT_POSITION_TOLERANCE),
                    ori_tol=target.get("rot_tol", DEFAULT_ORIENTATION_TOLERANCE),
                    max_vel_pos=target.get("max_vel_pos", None),
                    max_vel_ori=target.get("max_vel_ori", None),
                )
            else:
                print("[Waypoint - WARNING]: Ignoring target for unknown device: ", target["device"])

        self._target = TwoArmEETarget(left, right)

    def update(self, obs: OrderedDict):
        """
        Update the waypoint based on the current observation.
        :param obs: Current observation
        """
        raw_updates = self.__update_targets(obs)

        for update in raw_updates:
            if update["device"] == "robot_left":
                left_update = EETarget(
                    xyz=update["pos"],
                    quat=update["quat"],
                    grip=update["grip"],
                    pos_tol=update.get("pos_tol", DEFAULT_POSITION_TOLERANCE),
                    ori_tol=update.get("rot_tol", DEFAULT_ORIENTATION_TOLERANCE),
                    max_vel_pos=update.get("max_vel_pos", None),
                    max_vel_ori=update.get("max_vel_ori", None)
                )
                self._target.left = left_update
            elif update["device"] == "robot_right":
                right_update = EETarget(
                    xyz=update["pos"],
                    quat=update["quat"],
                    grip=update["grip"],
                    pos_tol=update.get("pos_tol", DEFAULT_POSITION_TOLERANCE),
                    ori_tol=update.get("rot_tol", DEFAULT_ORIENTATION_TOLERANCE),
                    max_vel_pos=update.get("max_vel_pos", None),
                    max_vel_ori=update.get("max_vel_ori", None),
                )
                self._target.right = right_update
            else:
                print("[Waypoint - WARNING]: Ignoring update for unknown device: ", update["device"])

    @property
    def target(self) -> TwoArmEETarget:
        """
        :return: Targets
        """
        return self._target

    def is_reached_by(
            self,
            current_robot_state: OrderedDict,
            env_config: str,
            dt: float) -> tuple[bool, bool]:
        """
        Check if the waypoint is reached by the current state.
        :param current_robot_state: Current state
        :param env_config: environment configuration ("single-robot", "parallel" or "opposed")
        :param dt: elapsed time in seconds
        :return: True if the waypoint is reached by the current state
        """
        unreachable = False
        if dt < self._min_duration:
            return False, False
        if dt > self._max_duration and self._must_reach:
            unreachable = True
        if dt >self._max_duration and not self._must_reach:
            return True, False

        current_robot_state = TwoArmEEState.from_dict(current_robot_state, env_config)
        is_reached = self._target.is_reached_by(current_robot_state)
        return is_reached, unreachable

    @property
    def id(self) -> int:
        """
        :return: ID
        """
        return self._id

    @property
    def description(self) -> str:
        """
        :return: Name
        """
        return self._des

    @property
    def uses_feedback(self) -> bool:
        """
        :return: True if the waypoint uses feedback
        """
        return self._uses_feedback

    @property
    def min_duration(self) -> float:
        """
        :return: Minimum duration before the waypoint is considered reached
        """
        return self._min_duration

    @property
    def max_duration(self) -> float:
        """
        :return: Maximum duration, after which the waypoint is considered unreachable
        """
        return self._max_duration

    def __create_targets(
            self,
            waypoint_dict: dict,
            known_waypoints: list,
            obs: OrderedDict = None
    ) -> list[dict]:
        """
        Create the list of targets for a waypoint.
        :param waypoint_dict:
        :param known_waypoints: List of known/already computed waypoints
        :param obs: Observation after environment reset
        :return:
        """

        mapped_targets = []
        for raw_target in waypoint_dict["targets"]:
            mapped_target = {
                "device": raw_target["device"],
                "pos_tol": raw_target.get("pos_tol", DEFAULT_POSITION_TOLERANCE),
                "rot_tol": raw_target.get("rot_tol", DEFAULT_ORIENTATION_TOLERANCE),
                "max_vel_pos": raw_target.get("max_vel_pos", None),
                "max_vel_ori": raw_target.get("max_vel_ori", None)
            }

            ####################################
            # 1 Check if ee_target is declared #
            ####################################
            ee_target = raw_target.get("ee_target", "")
            if ee_target:
                pattern = r"wp_(\d+)"
                if re.compile(pattern).match(ee_target):
                    previous_wp_id = int(ee_target.split("_")[1])
                    pos_quat_grip = self.__get_pos_quat_grip_from_previous_waypoint(raw_target["device"],
                                                                                    known_waypoints, previous_wp_id)
                else:
                    ee_target_method = self._waypoint_expert.ee_target_methods.get(ee_target, None)
                    if ee_target_method is None:
                        raise NotImplementedError(
                            f"[WP] Method {ee_target} not implemented in {self._waypoint_expert.__class__.__name__}")

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
                known_waypoints,
            )
            mapped_target["quat"] = self.__map_orientation(
                raw_target.get("quat", None),
                raw_target.get("euler", None),
                raw_target.get("ax_angle", None),
                raw_target["device"],
                known_waypoints,
            )
            mapped_target["grip"] = self.__map_gripper_target(raw_target.get("grip", None))

            mapped_targets.append(mapped_target)

        return mapped_targets

    def __update_targets(
            self,
            obs: OrderedDict = None
    ) -> list[dict]:
        """
        Create the list of targets for a waypoint.
        :param obs: Current observation after environment reset
        :return:
        """

        mapped_targets = []
        for raw_target in self._waypoint_dict["targets"]:
            mapped_target = {
                "device": raw_target["device"],
                "pos_tol": raw_target.get("pos_tol", DEFAULT_POSITION_TOLERANCE),
                "rot_tol": raw_target.get("rot_tol", DEFAULT_ORIENTATION_TOLERANCE),
                "max_vel_pos": raw_target.get("max_vel_pos", None),
                "max_vel_ori": raw_target.get("max_vel_ori", None)
            }

            ####################################
            # 1 Check if ee_target is declared #
            ####################################
            ee_target = raw_target.get("ee_target", "")
            if ee_target:
                pattern = r"wp_(\d+)"
                if re.compile(pattern).match(ee_target):
                    print("[Waypoint - INFO] Ignoring update request, only applicable for explicit ee_target_method.")
                    continue
                else:
                    ee_target_method = self._waypoint_expert.ee_target_methods.get(ee_target, None)
                    if ee_target_method is None:
                        raise NotImplementedError(
                            f"[WP] Method {ee_target} not implemented in {self._waypoint_expert.__class__.__name__}")

                    pos_quat_grip = ee_target_method(obs)

                mapped_target["pos"] = pos_quat_grip["pos"]
                mapped_target["quat"] = pos_quat_grip["quat"]
                mapped_target["grip"] = pos_quat_grip["grip"]

                mapped_targets.append(mapped_target)
                continue

            print("[Waypoint - INFO] Ignoring update request, only applicable for explicit ee_target_method.")

        return mapped_targets

    @staticmethod
    def __get_pos_quat_grip_from_previous_waypoint(device: str, known_waypoints: list, previous_wp_id: int) -> dict:
        """
        Get the position, orientation, and gripper state of a previous waypoint.
        :param device: Device name that the waypoint belongs to
        :param known_waypoints: List of previous waypoints
        :param previous_wp_id: ID of the previous waypoint
        :return: Dictionary containing the position, orientation, and gripper state
        """
        # find the previous waypoint by id
        known_wp = next(filter(lambda wp: wp.id == previous_wp_id, known_waypoints), None)

        if known_wp is None:
            raise ValueError(f"[WP] Previous waypoint with id {previous_wp_id} not found. Invalid yaml configuration.")

        previous_target = known_wp.target.left if device == "robot_left" else known_wp.target.right
        return {
            "pos": previous_target.xyz,
            "quat": previous_target.quat,
            "grip": previous_target.grip
        }

    def __map_position(self, pos: list | str | None, device: str, known_waypoints: list) -> np.array:
        """
        Map the position from a list to the target position.
        :param pos: Position as a list or string that needs to be mapped ("wp_<id>" or "wp_<id> + [dx, dy, dz]")
        :param device: device name that the position belongs to
        :param known_waypoints: List of previous waypoints
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
                pos_quat_grip = self.__get_pos_quat_grip_from_previous_waypoint(device, known_waypoints, previous_wp_id)
                return pos_quat_grip["pos"]

            # "wp_<id> + [dx, dy, dz]"
            pattern = r"wp_(\d+)\s*\+\s*\[(-?\d*\.?\d+(?:,\s*-?\d*\.?\d+)*)\]"
            match = re.match(pattern, pos)
            if match:
                previous_wp_id = int(match.group(1))
                d_xyz = np.array([float(x) for x in match.group(2).split(",")])
                if len(d_xyz) != 3:
                    raise ValueError(f"[WP] Invalid position string {pos}. Expected 3 values for [dx, dy, dz]")
                previous_pos_quat_grip = self.__get_pos_quat_grip_from_previous_waypoint(device, known_waypoints,
                                                                                         previous_wp_id)
                return previous_pos_quat_grip["pos"] + d_xyz

            # invalid string
            raise ValueError(f"[WP] Invalid position string {pos}. Expected 'wp_<id>' or 'wp_<id> + [dx, dy, dz]'")

        raise ValueError(f"[WP] Invalid position. Either 'pos' must be a list or a string.")

    def __map_orientation(
            self,
            quat: list | str | None,
            euler: list | str | None,
            ax_angle: list | str | None,
            device: str, previous_waypoints: list
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
            known_waypoints: list,
            num_values: int = 4,
            map_fn: callable = lambda x: x
    ) -> np.array:
        """
        Map the quaternion from a list to the target quaternion.
        :param rot: Orientation as a list
            or string that needs to be mapped ("wp_<id>" or "wp_<id> * [...]" or "[...] * wp_<id>"),
            where [...] is a list of num_values floats
        :param device: device name that the orientation belongs to
        :param known_waypoints: List of previous waypoints
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
                pos_quat_grip = self.__get_pos_quat_grip_from_previous_waypoint(device, known_waypoints, previous_wp_id)
                return pos_quat_grip["quat"]

            # "wp_<id> * [x, y, z, w]"
            pattern = r"wp_(\d+)\s*\*\s*\[(-?\d*\.?\d+(?:,\s*-?\d*\.?\d+)*)\]"
            match = re.match(pattern, rot)
            if match:
                previous_wp_id = int(match.group(1))
                d_rot = np.array([float(x) for x in match.group(2).split(",")])
                if len(d_rot) != num_values:
                    raise ValueError(
                        f"[WP] Invalid rotation string {rot}. Expected {num_values} values, got {len(d_rot)}")
                d_quat = map_fn(d_rot)
                prev_quat = self.__get_pos_quat_grip_from_previous_waypoint(device, known_waypoints, previous_wp_id)[
                    "quat"]
                return quat_multiply(prev_quat, d_quat)

            # "[x, y, z, w] * wp_<id>"
            pattern = r"\[(-?\d*\.?\d+(?:,\s*-?\d*\.?\d+)*)\]\s*\*\s*wp_(\d+)"
            match = re.match(pattern, rot)
            if match:
                d_rot = np.array([float(x) for x in match.group(1).split(",")])
                if len(d_rot) != num_values:
                    raise ValueError(
                        f"[WP] Invalid rotation string {rot}. Expected {num_values} values, got {len(d_rot)}")
                previous_wp_id = int(match.group(2))
                d_quat = map_fn(d_rot)
                pre_quat = self.__get_pos_quat_grip_from_previous_waypoint(device, known_waypoints, previous_wp_id)[
                    "quat"]
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





