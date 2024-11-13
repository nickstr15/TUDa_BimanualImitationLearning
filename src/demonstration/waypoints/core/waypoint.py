import numpy as np
from collections import OrderedDict

from src.utils.robot_states import TwoArmEEState
from src.utils.robot_targets import EETarget, TwoArmEETarget, GripperTarget

DEFAULT_POSITION_TOLERANCE = 0.01
DEFAULT_ORIENTATION_TOLERANCE = np.deg2rad(5)
DEFAULT_MIN_DURATION = 0.5 # before this time, the waypoint is not considered reached
DEFAULT_MAX_DURATION = 30.0 # after this time, the waypoint is considered as unreachable
DEFAULT_MUST_REACH = True

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
        self._des = waypoint_dict["description"]
        self._min_duration = waypoint_dict.get("min_duration", DEFAULT_MIN_DURATION)
        self._max_duration = waypoint_dict.get("max_duration", DEFAULT_MAX_DURATION)
        self._must_reach = waypoint_dict.get("must_reach", DEFAULT_MUST_REACH)

        left = EETarget()
        right = EETarget()


        for target in waypoint_dict["targets"]:
            if target["device"] == "robot_left":
                left = EETarget(
                    xyz=target["pos"],
                    quat=target["quat"],
                    grip=target["grip"],
                    pos_tol=target.get("pos_tol", DEFAULT_POSITION_TOLERANCE),
                    ori_tol=target.get("rot_tol", DEFAULT_ORIENTATION_TOLERANCE)
                )
            elif target["device"] == "robot_right":
                right = EETarget(
                    xyz=target["pos"],
                    quat=target["quat"],
                    grip=target["grip"],
                    pos_tol=target.get("pos_tol", DEFAULT_POSITION_TOLERANCE),
                    ori_tol=target.get("rot_tol", DEFAULT_ORIENTATION_TOLERANCE)
                )
            else:
                print("[Waypoint - WARNING]: Ignoring target for unknown device: ", target["device"])

        self._target = TwoArmEETarget(left, right)

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





