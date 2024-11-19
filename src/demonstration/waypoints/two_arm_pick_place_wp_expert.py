from typing import OrderedDict
import numpy as np

import robosuite as suite
from robosuite.utils.transform_utils import quat2mat, mat2euler, axisangle2quat, quat_multiply

from src.demonstration.waypoints.two_arm_handover_wp_expert import TwoArmHandoverWaypointExpert
from src.environments import TwoArmPickPlace
from src.utils.robot_targets import GripperTarget


class TwoArmPickPlaceWaypointExpert(TwoArmHandoverWaypointExpert):
    """
    Specific waypoint expert for the TwoArmPickPlace environment.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._env : TwoArmPickPlace = self._env # for type hinting in pycharm

        self._target_bin_name = "bin"
        self._get_hammer_head_halfsize_fn = lambda: self._env.hammer.head_halfsize

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
            "pre_drop_off_hammer_left": self.__pre_drop_off_hammer_left,
            "drop_off_hammer_left": self.__drop_off_hammer_left,
        }

        dct.update(update)
        return dct

    def __pre_drop_off_hammer_left(self, obs: OrderedDict = None) -> dict:
        """
        Pre-drop off position for the left arm.

        :param obs: observation after reset
        :return: dictionary with the target position, orientation, and gripper state
        """
        return self.__drop_off_computation(
            obs=obs,
            z_offset=self._bin_height + 0.2
        )

    def __drop_off_hammer_left(self, obs: OrderedDict = None) -> dict:
        """
        Drop off position for the left arm.

        :param obs: observation after reset
        :return: dictionary with the target position, orientation, and gripper state
        """
        head_halfsize = self._get_hammer_head_halfsize_fn()
        return self.__drop_off_computation(
            obs=obs,
            z_offset= self._bin_height/2 + head_halfsize + 0.03
        )

    def __drop_off_computation(self, obs: OrderedDict = None, z_offset: int = 0) -> dict:
        """
        Drop off position for the left arm.
        :param obs: observation after reset
        :param z_offset: offset in z direction
        :return: dictionary with the target position, orientation, and gripper state
        """

        """
        # recompute grip offset arm0
        hammer_quat = obs[f"{self._hammer_name}_quat"]
        flip_sign = np.sign(mat2euler(quat2mat(hammer_quat))[0])
        handle_length = self._get_hammer_handle_length_fn()
        grip_offset_0 = 0.5 * handle_length - 0.02  # grasp 2cm from hammer head
        grip_offset_0 *= -1 * flip_sign * self._handover_mode

        # compute grip offset arm1, 6cm shifted to handle center from grip offset arm0
        grip_offset_1 = grip_offset_0 - 0.06 * np.sign(grip_offset_0)
        """

        dct = self._calculate_target_pose(
            obj_pos=obs[f"{self._target_bin_name}_pos"],
            obj_quat=obs.get(f"{self._target_bin_name}_quat", np.array([0, 0, 0, 1])),
            offset=np.array([0.0, 0.0, z_offset]),
            gripper_state=GripperTarget.CLOSED_VALUE,
            adjustment_quat=axisangle2quat(np.array([0, 0, 1]) * np.pi/4)
        )
        return dct

def example(
    n_episodes: int = 10,
    robots: str | list[str] = ["Panda"]*2
):
    two_arm_pick_place = suite.make(
        env_name="TwoArmPickPlace",
        robots=robots,
        env_configuration="parallel",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )

    expert = TwoArmPickPlaceWaypointExpert(
        environment=two_arm_pick_place,
        waypoints_file="two_arm_pick_place_wp.yaml",
    )
    expert.visualize(n_episodes)


if __name__ == "__main__":
    example()
    #example(2, ["Kinova3", "Kinova3"])
    #example(2, ["IIWA", "IIWA"])
    #example(2, ["UR5e", "UR5e"])
    #example(2, ["Panda", "IIWA"])