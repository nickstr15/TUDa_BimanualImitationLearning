from typing import OrderedDict
import numpy as np

import robosuite as suite
from robosuite import TwoArmPegInHole
from robosuite.utils.transform_utils import quat2mat, mat2euler, axisangle2quat, quat_multiply, euler2mat, mat2quat

from src.utils.robot_targets import GripperTarget
from src.demonstration.waypoints.core.waypoint_expert import TwoArmWaypointExpertBase


class TwoArmPegInHoleWaypointExpert(TwoArmWaypointExpertBase):
    """
    Specific waypoint expert for the TwoArmPegInHole environment.
    """
    target_env_name = "TwoArmPegInHole"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._env: TwoArmPegInHole = self._env  # for type hinting in pycharm

        self._hole_offset = np.array([0, 0, -0.17]) # offset for the hole from ee

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
            "pre_target_left": self.__pre_target_left,
            "pre_target_right": self.__pre_target_right,

            "target_left": self.__target_left,
            "target_right": self.__target_right,
        }

        dct.update(update)
        return dct

    def __pre_target_left(self, obs: OrderedDict = None) -> dict:
        """
        Random target for the left arm.
        :param obs:
        :return:
        """
        pos = np.array([-0.2, 0.2, 1.2])
        quat = self._null_quat_left

        return {
            "pos": pos,
            "quat": quat,
            "grip": GripperTarget.OPEN_VALUE #grip irrelevant
        }

    def __pre_target_right(self, obs: OrderedDict = None) -> dict:
        """
        Pre-target position for the right arm.
        :param obs:
        :return:
        """
        pos = np.array([-0.2, -(0.2+self._env.peg_length), 1.2]) + self._hole_offset
        quat = quat_multiply(axisangle2quat(np.array([1, 0, 0]) * np.pi/2), self._null_quat_right)

        return {
            "pos": pos,
            "quat": quat,
            "grip": GripperTarget.OPEN_VALUE #grip irrelevant
        }

    def __target_left(self, obs: OrderedDict = None) -> dict:
        """
        Target position for the left arm.
        :param obs:
        :return:
        """
        pos = np.array([-0.2, 0, 1.2])
        quat = self._null_quat_left

        return {
            "pos": pos,
            "quat": quat,
            "grip": GripperTarget.OPEN_VALUE #grip irrelevant
        }

    def __target_right(self, obs: OrderedDict = None) -> dict:
        """
        Target position for the right arm.
        :param obs:
        :return:
        """
        pos = np.array([-0.2, -self._env.peg_length, 1.2]) + self._hole_offset
        quat = quat_multiply(axisangle2quat(np.array([1, 0, 0]) * np.pi/2), self._null_quat_right)

        return {
            "pos": pos,
            "quat": quat,
            "grip": GripperTarget.OPEN_VALUE #grip irrelevant
        }

if __name__ == "__main__":
    expert = TwoArmPegInHoleWaypointExpert
    f = "two_arm_peg_in_hole_wp.yaml"

    expert.example(f, num_recording_episodes=1)
    #expert.example(f, robots=["Kinova3"]*2)
    #expert.example(f, robots=["IIWA"]*2)
    #expert.example(f, robots=["UR5e"]*2, gripper_types=["Robotiq85Gripper"]*2)
    # expert.example(f, robots=["Panda", "IIWA"])