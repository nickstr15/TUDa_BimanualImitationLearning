from typing import Dict

import numpy as np
from transforms3d.quaternions import qmult, qinverse, quat2axangle

from src.control.utils.arm_state import ArmState
from src.control.utils.enums import GripperState
from src.utils.constants import MAX_DELTA_TRANSLATION, MAX_DELTA_ROTATION

DEFAULT_POSITION_TOLERANCE = 0.01
DEFAULT_ORIENTATION_TOLERANCE = np.deg2rad(5)
DEFAULT_MIN_DURATION = 1.0
DEFAULT_MAX_DURATION = 10.0

class ArmStateTarget(ArmState):
    """
    The ArmStateTarget class extends the ArmState class
    by adding tolerance values for the position and orientation.
    """

    def __init__(
        self,
        xyz: np.ndarray = np.zeros(3),
        rot: np.ndarray = np.zeros(3),
        grip: GripperState = GripperState.OPEN,
        position_tolerance: float = DEFAULT_POSITION_TOLERANCE,
        orientation_tolerance: float = DEFAULT_ORIENTATION_TOLERANCE,
    ) -> None:
        """
        :param xyz: xyz position
        :param rot: euler angles or quaternion
        :param grip: gripper state (open or closed)
        :param position_tolerance: threshold quantifying if pos_target is reached
        :param orientation_tolerance: threshold quantifying if tor_target is reached
        """
        super().__init__(xyz, rot, grip)
        self._position_tolerance = position_tolerance
        self._orientation_tolerance = orientation_tolerance


    def get_position_tolerance(self) -> float:
        """
        :return: Position tolerance
        """
        return self._position_tolerance

    def get_orientation_tolerance(self) -> float:
        """
        :return: Orientation tolerance
        """
        return self._orientation_tolerance

    def is_reached_by(self, current_state: ArmState) -> bool:
        """
        Check if the target state is reached by the current state.
        :param current_state: Current state
        :return: True if the target state is reached by the current state
        """
        current_xyz = current_state.get_xyz()
        current_quat = current_state.get_quat()
        current_grip = current_state.get_gripper_state()

        target_xyz = self.get_xyz()
        target_quat = self.get_quat()
        target_grip = self.get_gripper_state()

        position_diff = np.linalg.norm(target_xyz - current_xyz)
        orientation_diff = quat2axangle(qmult(current_quat, qinverse(target_quat)))[1]

        return position_diff <= self._position_tolerance \
            and orientation_diff <= self._orientation_tolerance \
            and current_grip == target_grip

class Waypoint:
    """
    A class to represent a waypoint.
    """

    def __init__(self, waypoint_dict: dict) -> None:
        """
        Constructor for the Waypoint class.
        :param waypoint_dict: Dictionary containing the waypoint data
        """
        self._load_waypoint(waypoint_dict)

    def _load_waypoint(self, waypoint_dict: dict) -> None:
        """
        Load the waypoint data from the dictionary.
        :param waypoint_dict:
        :return:
        """
        self._id = waypoint_dict["id"]
        self._name = waypoint_dict["name"]
        self._min_duration = waypoint_dict.get("min_duration", DEFAULT_MIN_DURATION)
        self._max_duration = waypoint_dict.get("max_duration", DEFAULT_MAX_DURATION)
        self._max_delta_translation = waypoint_dict.get("max_delta_translation", None)
        self._max_delta_rotation = waypoint_dict.get("max_delta_rotation", None)
        self._targets = {}
        for target in waypoint_dict["targets"]:
            self._targets[target["device"]] = ArmStateTarget(
                xyz=target["position"],
                rot=target["orientation"],
                grip=target["gripper"],
                position_tolerance=target.get("position_tolerance", DEFAULT_POSITION_TOLERANCE),
                orientation_tolerance=target.get("orientation_tolerance", DEFAULT_ORIENTATION_TOLERANCE)
            )

    @property
    def targets(self) -> Dict[str, ArmStateTarget]:
        """
        :return: Targets
        """
        return self._targets

    def is_reached_by(self, current_robot_state: dict[str, ArmState], dt : float) -> bool:
        """
        Check if the waypoint is reached by the current state.
        :param current_robot_state: Current state
        :param dt: elapsed time in seconds
        :return: True if the waypoint is reached by the current state
        """
        if dt < self._min_duration:
            return False
        if dt > self._max_duration:
            return True

        is_reached = True
        for device, target in self._targets.items():
            is_reached = is_reached and target.is_reached_by(current_robot_state[device])

        return is_reached

    @property
    def id(self) -> int:
        """
        :return: ID
        """
        return self._id

    @property
    def name(self) -> str:
        """
        :return: Name
        """
        return self._name

    @property
    def min_duration(self) -> float:
        """
        :return: Minimum duration
        """
        return self._min_duration

    @property
    def max_duration(self) -> float:
        """
        :return: Maximum duration
        """
        return self._max_duration

    @property
    def max_delta_translation(self) -> float | None:
        """
        :return: Maximum delta translation
        """
        return self._max_delta_translation

    @property
    def max_delta_rotation(self) -> float | None:
        """
        :return: Maximum delta rotation
        """
        return self._max_delta_rotation


