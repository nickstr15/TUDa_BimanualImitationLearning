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

        self._target_pos_left = np.array([0.0, 0.0, 1.0])
        self._target_quat_left = self._null_quat_left

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
            "random_target_left": self.__random_target_left,
            "pre_target_right": self.__pre_target_right,
        }

        dct.update(update)
        return dct

    def __random_target_left(self, obs: OrderedDict = None) -> dict:
        """
        Random target for the left arm.
        :param obs:
        :return:
        """
        euler_x = np.random.uniform(np.deg2rad(-30), 0)
        euler_y = np.random.uniform(np.deg2rad(20), np.deg2rad(20))
        euler_z = np.random.uniform(np.deg2rad(10), np.deg2rad(10))
        random_quat = mat2quat(euler2mat(np.array([euler_x, euler_y, euler_z])))

        pos_x = np.random.uniform(-0.1, 0.1)
        pos_y = np.random.uniform(0.1, 0.3)
        pos_z = np.random.uniform(1.0, 1.2)
        random_pos = np.array([pos_x, pos_y, pos_z])

        self._target_quat_left = random_quat
        self._target_pos_left = random_pos

        return {
            "pos": random_pos,
            "quat": quat_multiply(random_quat, self._null_quat_left),
            "grip": GripperTarget.OPEN_VALUE #grip irrelevant
        }

    def __pre_target_right(self, obs: OrderedDict = None) -> dict:
        """
        Pre-target position for the right arm.
        :param obs:
        :return:
        """
        raise NotImplementedError # TODO

def example(
    n_episodes: int = 10,
    robots: str | list[str] = ["Panda"]*2
):
    two_arm_pick_place = suite.make(
        env_name="TwoArmPegInHole",
        robots=robots,
        env_configuration="parallel",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )

    expert = TwoArmPegInHoleWaypointExpert(
        environment=two_arm_pick_place,
        waypoints_file="two_arm_peg_in_hole_wp.yaml",
    )
    expert.visualize(n_episodes)


if __name__ == "__main__":
    example()
    #example(2, ["Kinova3", "Kinova3"])
    #example(2, ["IIWA", "IIWA"])
    #example(2, ["UR5e", "UR5e"])
    #example(2, ["Panda", "IIWA"])