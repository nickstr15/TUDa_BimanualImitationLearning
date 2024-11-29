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

def example(
    num_episodes: int = 10,
    robots: str | list[str] = ["Panda"]*2,
    num_recording_episodes: int = 0,
):
    env = suite.make(
        env_name="TwoArmPegInHole",
        robots=robots,
        env_configuration="parallel",
        has_renderer=True,
        has_offscreen_renderer=num_recording_episodes > 0,
        use_camera_obs=num_recording_episodes > 0,
    )

    expert = TwoArmPegInHoleWaypointExpert(
        environment=env,
        waypoints_file="two_arm_peg_in_hole_wp.yaml",
    )
    expert.visualize(
        num_episodes=num_episodes,
        num_recording_episodes=num_recording_episodes,
    )


if __name__ == "__main__":
    example()
    #example(2, ["Kinova3", "Kinova3"])
    #example(2, ["IIWA", "IIWA"])
    #example(2, ["UR5e", "UR5e"], ["Robotiq85Gripper", "Robotiq85Gripper"])
    #example(2, ["Panda", "IIWA"])