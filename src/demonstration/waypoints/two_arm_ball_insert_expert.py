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
    target_env_name = "TwoArmBallInsert"

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

            "drop_right": lambda obs: self.__drop(obs, "right"),
            "drop_left": lambda obs: self.__drop(obs, "left"),
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
        ball_pos_target = bin_pos
        target_height = self._env.bin.bin_size[2] / 2 + self._env.ball.size[0] + 0.05
        ball_pos_target[2] += target_height

        # vector from ball to ball_target
        v = ball_pos_target - ball_pos

        two_arm_ee_state = TwoArmEEState.from_dict(obs, self._env.env_configuration)
        ee_state = two_arm_ee_state.right if arm == "right" else two_arm_ee_state.left

        current_pos = ee_state.xyz
        current_quat = ee_state.quat

        current_pos[0] += v[0]
        current_pos[1] += v[1]

        return {
            "pos": current_pos,
            "quat": current_quat,
            "grip": GripperTarget.OPEN_VALUE,
        }

    def __drop(self, obs: OrderedDict, arm: str) -> dict:
        """
        Get the target to drop the ball into the bin.
        :param obs: current observation
        :param arm: the arm to get the target for
        :return: the target
        """
        assert arm in ["right", "left"], f"Invalid arm: {arm}"

        two_arm_ee_state = TwoArmEEState.from_dict(obs, self._env.env_configuration)
        ee_state = two_arm_ee_state.right if arm == "right" else two_arm_ee_state.left

        sign = -1 if arm == "right" else 1

        current_pos = ee_state.xyz
        current_quat = ee_state.quat

        current_pos[1] += sign * 0.2

        return {
            "pos": current_pos,
            "quat": current_quat,
            "grip": GripperTarget.OPEN_VALUE,
        }

if __name__ == "__main__":
    expert = TwoArmBallInsertWaypointExpert
    f = "two_arm_ball_insert_wp.yaml"

    expert.example(f, num_recording_episodes=1)
    #expert.example(f, robots=["Kinova3"]*2)
    #expert.example(f, robots=["IIWA"]*2)
    #expert.example(f, robots=["UR5e"]*2, gripper_types=["Robotiq85Gripper"]*2)
    #expert.example(f, robots=["Panda", "IIWA"])