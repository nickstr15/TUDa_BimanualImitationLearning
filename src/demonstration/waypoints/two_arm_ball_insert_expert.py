from typing import OrderedDict
import numpy as np
from copy import deepcopy

import robosuite as suite
from robosuite.utils.transform_utils import quat2mat, mat2euler, axisangle2quat, quat_multiply

from src.environments.manipulation.two_arm_ball_insert import TwoArmBallInsert
from src.utils.robot_states import TwoArmEEState
from src.utils.robot_targets import GripperTarget
from src.demonstration.waypoints.core.waypoint_expert import TwoArmWaypointExpertBase


class TwoArmBallInsertWaypointExpert(TwoArmWaypointExpertBase):
    """
    Specific waypoint expert for the TwoArmLift environment.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._env: TwoArmBallInsert = self._env  # for type hinting in pycharm

        self._pre_pre_pickup_right_dict = None
        self._pre_pre_pickup_left_dict = None

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
            "pre_pre_pickup_right": self.__pre_pre_pickup_right,
            "pre_pre_pickup_left": self.__pre_pre_pre_pickup_left,

            "pre_pickup_right": self.__pre_pickup_right,
            "pre_pickup_left": self.__pre_pickup_left,

            "above_bin_right": lambda obs: self.__above_bin(obs, "right"),
            "above_bin_left": lambda obs: self.__above_bin(obs, "left"),
        }

        dct.update(update)
        return dct

    def __pre_pre_pickup_right(self, obs: OrderedDict) -> dict:
        """
        Get the pre-pre-pickup target for the right arm.
        => Rotates the arm into the correct orientation to pick up the ball.

        :param obs: the observation after reset
        :return: the target
        """
        ball_pos = obs["ball_pos"]
        target_pos = ball_pos + np.array([
            0.0,
            -(self._env.ball.size[0]+0.2),
            -0.75*self._env.ball.size[0]
        ])

        dq1 = axisangle2quat(np.array([0, 0, 1]) * np.deg2rad(90))
        dq2 = axisangle2quat(np.array([1, 0, 0]) * np.deg2rad(60))
        target_quat = quat_multiply(dq2, quat_multiply(dq1, self._null_quat_right))

        self._pre_pre_pickup_right_dict = {
            "pos": target_pos,
            "quat": target_quat,
            "grip": GripperTarget.OPEN_VALUE,
        }

        return self._pre_pre_pickup_right_dict


    def __pre_pre_pre_pickup_left(self, obs: OrderedDict) -> dict:
        """
        Get the pre-pre-pickup target for the left arm.
        => Rotates the arm into the correct orientation to pick up the ball.

        :param obs: the observation after reset
        :return: the target
        """
        ball_pos = obs["ball_pos"]
        target_pos = ball_pos + np.array([
            0.0,
            (self._env.ball.size[0]+0.2),
            -0.75*self._env.ball.size[0]
        ])

        dq1 = axisangle2quat(np.array([0, 0, 1]) * np.deg2rad(90))
        dq2 = axisangle2quat(np.array([-1, 0, 0]) * np.deg2rad(60))
        target_quat = quat_multiply(dq2, quat_multiply(dq1, self._null_quat_left))

        self._pre_pre_pickup_left_dict = {
            "pos": target_pos,
            "quat": target_quat,
            "grip": GripperTarget.OPEN_VALUE,
        }

        return self._pre_pre_pickup_left_dict

    def __pre_pickup_right(self, obs: OrderedDict) -> dict:
        """
        Get the pre-pickup target for the right arm.
        => Moves the arm to the correct position to pick up the ball.

        :param obs: the observation after reset
        :return: the target
        """
        target_y = -1*(np.sqrt(1 - 0.75**2) * self._env.ball.size[0])

        dct = deepcopy(self._pre_pre_pickup_right_dict)
        dct["pos"][1] = target_y

        return dct

    def __pre_pickup_left(self, obs: OrderedDict) -> dict:
        """
        Get the pre-pickup target for the left arm.
        => Moves the arm to the correct position to pick up the ball.

        :param obs: the observation after reset
        :return: the target
        """
        target_y = np.sqrt(1 - 0.75**2) * self._env.ball.size[0]

        dct = deepcopy(self._pre_pre_pickup_left_dict)
        dct["pos"][1] = target_y

        return dct

    def __above_bin(self, obs: OrderedDict, arm: str) -> dict:
        """
        Get the target to move the arm above the bin.

        :param obs: the observation after reset
        :param arm: the arm to get the target for
        :return: the target
        """
        assert arm in ["right", "left"], f"Invalid arm: {arm}"

        ball_pos = obs["ball_pos"]

        bin_pos = obs["bin_pos"]
        ball_pos_target = bin_pos + self._env.bin.bin_size[2]/2 + self._env.ball.size[0] + 0.05

        #vector from ball to ball_target
        v = ball_pos_target - ball_pos

        two_arm_ee_state = TwoArmEEState.from_dict(obs, self._env.env_configuration)
        ee_state = two_arm_ee_state.right if arm == "right" else two_arm_ee_state.left

        current_pos = ee_state.xyz
        current_quat = ee_state.quat

        current_pos[0] += v[0]

        print(v)
        return {
            "pos": current_pos,
            "quat": current_quat,
            "grip": GripperTarget.OPEN_VALUE,
        }


def example(
    n_episodes: int = 10,
    robots: str | list[str] = ["Panda"]*2,
    gripper_types: str | list[str] = ["default", "default"]
):
    two_arm_pick_place = suite.make(
        env_name="TwoArmBallInsert",
        gripper_types=gripper_types,
        robots=robots,
        env_configuration="parallel",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )

    expert = TwoArmBallInsertWaypointExpert(
        environment=two_arm_pick_place,
        waypoints_file="two_arm_ball_insert_wp.yaml",
    )
    expert.visualize(n_episodes)


if __name__ == "__main__":
    example()
    #example(2, ["Kinova3", "Kinova3"])
    #example(2, ["IIWA", "IIWA"])
    #example(2, ["UR5e", "UR5e"], ["Robotiq85Gripper", "Robotiq85Gripper"])
    #example(2, ["Panda", "IIWA"])