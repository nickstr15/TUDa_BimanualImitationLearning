from typing import OrderedDict

import numpy as np
import robosuite as suite
from robosuite.utils.transform_utils import quat2mat, quat2axisangle, mat2quat, quat_multiply, quat_inverse, \
    convert_quat, axisangle2quat

from src.demonstration.waypoints.two_arm_lift_wp_expert import TwoArmLiftWaypointExpert
from src.environments.manipulation.two_arm_quad_insert import TwoArmQuadInsert
from src.utils.robot_states import TwoArmEEState
from src.utils.robot_targets import GripperTarget


class TwoArmQuadInsertWaypointExpert(TwoArmLiftWaypointExpert):
    """
    Specific waypoint expert for the TwoArmLift environment.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._env: TwoArmQuadInsert = self._env  # for type hinting in pycharm

        self._object_name = "bracket"
        self._initial_insertion_offset = 0.05

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
            "pre_goal_robot_right": lambda obs: self.__compute_pre_goal(obs, "right"),
            "pre_goal_robot_left": lambda obs: self.__compute_pre_goal(obs, "left"),

            "goal_robot_right": lambda obs: self.__compute_target_pose(obs, "right"),
            "goal_robot_left": lambda obs: self.__compute_target_pose(obs, "left"),

            "goal_robot_right_1": lambda obs: self.__compute_target_pose(obs, "right", GripperTarget.OPEN_VALUE),
            "goal_robot_left_1": lambda obs: self.__compute_target_pose(obs, "left", GripperTarget.OPEN_VALUE),
        }

        dct.update(update)
        return dct

    def __compute_pre_goal(
            self,
            obs: OrderedDict,
            arm: str
    ) -> dict:
        """
        Compute the pre-goal target pose for the robot arm.
        :param obs:
        :param arm:
        :return:
        """
        assert arm in ["right", "left"], f"Invalid arm: {arm}"

        handle0_xpos = obs["handle0_xpos"]
        handle1_xpos = obs["handle1_xpos"]
        if arm == "right":
            v_to_handle = self._env.bracket.center_to_handle0 if handle0_xpos[1] < handle1_xpos[1] \
                else self._env.bracket.center_to_handle1
        else:
            v_to_handle = self._env.bracket.center_to_handle0 if handle0_xpos[1] >= handle1_xpos[1] \
                else self._env.bracket.center_to_handle1

        target_xpos = np.array(obs["target_xpos"])
        target_quat = np.array(obs["target_quat"])

        # convert to world frame
        v_to_handle = quat2mat(target_quat) @ v_to_handle

        target_xpos += v_to_handle

        return self._calculate_pose(
            xpos=target_xpos,
            quat=target_quat,
            null_quat=self._null_quat_right,
            offset_xpos=np.array([0, 0, self._initial_insertion_offset]),
            grip=GripperTarget.CLOSED_VALUE,
        )

    def __compute_target_pose(
            self,
            obs: OrderedDict,
            arm: str,
            grip: GripperTarget = GripperTarget.CLOSED_VALUE
    ) -> dict:
        """
        Compute the target pose for the robot arm.

        :param obs: current observation
        :param arm: arm to compute the target for, either "right" or "left"
        :param grip: desired gripper state
        :return: dictionary with the target position, orientation, and gripper state
        """
        assert arm in ["right", "left"], f"Invalid arm: {arm}"

        two_arm_ee_state = TwoArmEEState.from_dict(obs, self._env.env_configuration)
        ee_state = two_arm_ee_state.right if arm == "right" else two_arm_ee_state.left

        H_current = np.eye(4)
        H_current[:3, 3] = ee_state.xyz
        H_current[:3, :3] = quat2mat(ee_state.quat)

        bracket_pos = obs["bracket_xpos"]
        bracket_quat = obs["bracket_quat"]
        peg_pos = obs["target_xpos"]
        peg_quat = obs["target_quat"]

        T = self.__compute_transformation(bracket_pos, bracket_quat, peg_pos, peg_quat)

        H_target = T @ H_current
        target_xpos = H_target[:3, 3]
        target_quat = mat2quat(H_target[:3, :3])

        return {
            "pos": target_xpos,
            "quat": target_quat,
            "grip": grip
        }


    @staticmethod
    def __compute_transformation(
            bracket_pos: np.ndarray,
            bracket_quat: np.ndarray,
            peg_pos: np.ndarray,
            peg_quat: np.ndarray
    ) -> np.ndarray:
        """
        Compute the transformation to move the bracket to the peg.

        :param bracket_pos: position of the bracket
        :param bracket_quat: orientation of the bracket
        :param peg_pos: position of the peg
        :param peg_quat: orientation of the peg

        :return: transformation matrix
        """
        H_bracket = np.eye(4)
        H_bracket[:3, 3] = bracket_pos
        H_bracket[:3, :3] = quat2mat(bracket_quat)

        H_peg = np.eye(4)
        H_peg[:3, 3] = peg_pos
        H_peg[:3, :3] = quat2mat(peg_quat)

        T = np.linalg.inv(H_bracket) @ H_peg
        return T

def example(
    num_episodes: int = 10,
    robots: str | list[str] = ["Panda"]*2,
    gripper_types: str | list[str] = ["default", "default"],
    num_recording_episodes: int = 0,
):
    env = suite.make(
        env_name="TwoArmQuadInsert",
        gripper_types=gripper_types,
        robots=robots,
        env_configuration="parallel",
        has_renderer=True,
        has_offscreen_renderer=num_recording_episodes > 0,
        use_camera_obs=num_recording_episodes > 0,
    )

    expert = TwoArmQuadInsertWaypointExpert(
        environment=env,
        waypoints_file="two_arm_quad_insert_wp.yaml",
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
