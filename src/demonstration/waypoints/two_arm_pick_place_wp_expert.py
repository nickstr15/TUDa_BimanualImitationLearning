from typing import OrderedDict, Callable
import numpy as np

import robosuite as suite
from Xlib.Xcursorfont import target
from robosuite.utils.transform_utils import quat2mat, mat2euler, axisangle2quat, quat_multiply, euler2mat, mat2quat

from src.utils.robot_states import TwoArmEEState, EEState
from src.utils.robot_targets import GripperTarget
from src.demonstration.waypoints.core.waypoint_expert import TwoArmWaypointExpertBase
from src.environments.manipulation.two_arm_pick_place import TwoArmPickPlace


class TwoArmPickPlaceWaypointExpert(TwoArmWaypointExpertBase):
    """
    Specific waypoint expert for the TwoArmPickPlace environment.
    """

    ######################################################################
    # Definition of special ee targets ###################################
    ######################################################################
    def _create_ee_target_methods_dict(self) -> dict:
        """
        Create a dictionary of methods to get the special end-effector targets.

        All methods take the observation after reset as input.

        :return: dictionary with the methods
        """
        dct = super()._create_ee_target_methods_dict()
        update = {
            "pre_pick_up_robot_right": self.__pre_pick_up_robot_right,
            "pre_drop_off_robot_left": self.__pre_drop_off_robot_left,

            "hand_over_robot_right": self.__hand_over_robot_right,
            "pre_hand_over_robot_left": self.__pre_hand_over_robot_left,
        }

        dct.update(update)
        return dct

    def __pre_pick_up_robot_right(self, obs: OrderedDict = None) -> dict:
        """
        Pre-pick up position for the right arm.

        :param obs: observation after reset
        :return: dictionary with the target position, orientation, and gripper state
        """
        return self.__calculate_target_pose(
            obj_pos=obs["hammer_pos"],
            obj_quat=obs["hammer_quat"],
            offset=np.array([0.06, 0.0, -0.05]),
            gripper_state=GripperTarget.OPEN_VALUE
        )

    def __pre_drop_off_robot_left(self, obs: OrderedDict = None) -> dict:
        """
        Pre-drop off position for the left arm.

        :param obs: observation after reset
        :return: dictionary with the target position, orientation, and gripper state
        """
        return self.__calculate_target_pose(
            obj_pos=obs["bin_pos"],
            obj_quat=obs["bin_quat"],
            offset=np.array([-0.03, 0.0, -0.15]),
            gripper_state=GripperTarget.CLOSED_VALUE,
            angle_adjustment_fn=lambda a : np.abs(a)
        )

    def __calculate_target_pose(
            self,
            obj_pos: np.ndarray,
            obj_quat: np.ndarray,
            offset: np.ndarray = np.zeros(3),
            gripper_state: float = GripperTarget.OPEN_VALUE,
            angle_adjustment_fn: Callable[[float], float] = lambda x: x):
        """
        Calculates the target position, orientation, and gripper state based on the object's position and quaternion.

        :param obj_pos: Position of the object (e.g., hammer or bin)
        :param obj_quat: Quaternion orientation of the object
        :param offset: Offset to apply to the position
        :param gripper_state: Desired gripper state (e.g., open or closed)
        :param angle_adjustment_fn: Additional angle adjustment in degrees (e.g., -45 for diagonal drop-off)
        :return: Dictionary with the target position, orientation, and gripper state
        """
        # Calculate angle in degrees and map to symmetrical range
        angle_deg = -np.rad2deg(mat2euler(quat2mat(obj_quat))[0]) + 90
        if np.sign(mat2euler(quat2mat(obj_quat))[2]) < 0:
            angle_deg -= 180
        angle_deg = (angle_deg + 180) % 360 - 180
        angle_deg = angle_deg - 180 if angle_deg > 90 else (
            angle_deg + 180 if angle_deg < -90 else angle_deg)

        angle_deg = angle_adjustment_fn(angle_deg)

        # Apply angle adjustment and convert to quaternion
        angle_rad = np.deg2rad(angle_deg)
        target_quat = axisangle2quat(np.array([0, 0, 1]) * angle_rad)
        target_quat = quat_multiply(target_quat, self._null_quat_left)

        # Calculate target position with offset
        target_pos = obj_pos + np.dot(quat2mat(target_quat), offset)
        return {
            "pos": target_pos,
            "quat": target_quat,
            "grip": gripper_state
        }

    def __hand_over_robot_right(self, obs: OrderedDict = None) -> dict:
        """
        Hand-over position for the right arm.

        :param obs: observation after reset
        :return: dictionary with the target position, orientation, and gripper state
        """
        target_pos = np.array([0.03, 0.0, 1])
        target_quat = quat_multiply(
            axisangle2quat(np.array([1, 0, 0]) * np.deg2rad(90)),
            self._null_quat_right
        )
        return {
            "pos": target_pos,
            "quat": target_quat,
            "grip": GripperTarget.CLOSED_VALUE
        }

    def __pre_hand_over_robot_left(self, obs: OrderedDict = None) -> dict:
        target_pos = np.array([-0.03, 0, 1.1])
        target_quat = self._null_quat_right
        return {
            "pos": target_pos,
            "quat": target_quat,
            "grip": GripperTarget.OPEN_VALUE
        }


def example_panda(
    n_episodes: int = 10,
    robots: str | list[str] = ["Panda", "Panda"]
):
    two_arm_pick_place = suite.make(
        env_name="TwoArmPickPlace",
        robots=robots,
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )

    expert = TwoArmPickPlaceWaypointExpert(
        two_arm_pick_place,
        waypoints_file="two_arm_pick_place_wp.yaml",
        null_euler_left=[np.pi, 0, 0],
        null_euler_right=[np.pi, 0, 0]
    )
    expert.visualize(n_episodes)


if __name__ == "__main__":
    #example_panda(2, ["Panda", "Panda"])
    #example_panda(2, ["Kinova3", "Kinova3"])
    #example_panda(2, ["IIWA", "IIWA"])
    #example_panda(2, ["UR5e", "UR5e"])

    example_panda(2, ["Panda", "IIWA"])