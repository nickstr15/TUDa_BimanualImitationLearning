import numpy as np
from transforms3d.quaternions import qmult, qinverse, quat2axangle

from src.control.utils.arm_state import ArmState
from src.control.utils.enums import GripperState

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
        xyz_abg: np.ndarray = np.zeros(6),
        xyz_abg_vel: np.ndarray = np.zeros(6),
        grip: GripperState = GripperState.OPEN,
        position_tolerance: float = DEFAULT_POSITION_TOLERANCE,
        orientation_tolerance: float = DEFAULT_ORIENTATION_TOLERANCE
    ):
        super().__init__(xyz_abg, xyz_abg_vel, grip)
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
        orientation_diff = quat2axangle(qmult(current_quat, qinverse(target_quat)))[0][1],
        print(position_diff, orientation_diff, current_grip, target_grip)
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
        self._targets = {}
        for target in waypoint_dict["targets"]:
            self._targets[target["device"]] = ArmStateTarget(
                xyz_abg=target["position"] + target["orientation"],
                xyz_abg_vel=target.get("velocity", [0, 0, 0]) + target.get("angular_velocity", [0, 0, 0]),
                grip=target["gripper"],
                position_tolerance=target.get("position_tolerance", DEFAULT_POSITION_TOLERANCE),
                orientation_tolerance=target.get("orientation_tolerance", DEFAULT_ORIENTATION_TOLERANCE)
            )

    @property
    def targets(self) -> dict:
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


