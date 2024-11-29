from typing import OrderedDict
import numpy as np

import robosuite as suite
from robosuite import TwoArmLift
from robosuite.utils.transform_utils import quat2mat, mat2euler, axisangle2quat, quat_multiply

from src.utils.robot_targets import GripperTarget
from src.demonstration.waypoints.core.waypoint_expert import TwoArmWaypointExpertBase


class TwoArmLiftWaypointExpert(TwoArmWaypointExpertBase):
    """
    Specific waypoint expert for the TwoArmLift environment.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._env: TwoArmLift = self._env  # for type hinting in pycharm

        self._object_name = "pot"

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
            "pre_pick_up_robot_left": self.__pre_pick_up_robot_left,
            "pre_pick_up_robot_right": self.__pre_pick_up_robot_right,
        }

        dct.update(update)
        return dct

    def __pre_pick_up_robot_left(self, obs: OrderedDict = None) -> dict:
        """
        Pre-pick up position for the left arm.

        :param obs: observation after reset
        :return: dictionary with the target position, orientation, and gripper state
        """
        handle0_xpos = obs["handle0_xpos"]
        handle1_xpos = obs["handle1_xpos"]
        # select handle_xpos by the one with larger y value
        handle_xpos = handle0_xpos if handle0_xpos[1] >= handle1_xpos[1] else handle1_xpos
        object_quat = obs[f"{self._object_name}_quat"]

        return self._calculate_pose(
            xpos=handle_xpos,
            quat=object_quat,
            null_quat=self._null_quat_left,
        )

    def __pre_pick_up_robot_right(self, obs: OrderedDict = None) -> dict:
        """
        Pre-pick up position for the right arm.

        :param obs: observation after reset
        :return: dictionary with the target position, orientation, and gripper state
        """
        handle0_xpos = obs["handle0_xpos"]
        handle1_xpos = obs["handle1_xpos"]
        #select handle_xpos by the one with smaller y value
        handle_xpos = handle0_xpos if handle0_xpos[1] < handle1_xpos[1] else handle1_xpos
        object_quat = obs[f"{self._object_name}_quat"]

        return self._calculate_pose(
            xpos=handle_xpos,
            quat=object_quat,
            null_quat=self._null_quat_right
        )

    @staticmethod
    def _calculate_pose(
            xpos: np.ndarray,
            quat: np.ndarray,
            null_quat: np.ndarray,
            offset_xpos: np.ndarray = np.array([0.0, 0.0, 0.1]),
            grip: float = GripperTarget.OPEN_VALUE,
    ) -> dict:
        """
        Calculate the pre-pickup pose for the arm.

        :param xpos: position of the handle
        :param quat: orientation of the pot
        :param null_quat: null orientation for the end-effector
        :param offset_xpos: offset for the position in world frame (!)
        :param grip: gripper state
        :return: dictionary with the target position, orientation, and gripper state
        """
        # compute the offset
        angle_deg = np.rad2deg(mat2euler(quat2mat(quat)))[2]

        # map angle to +-180°
        angle_deg = (angle_deg + 180) % 360 - 180

        # due to symmetry map angle to +-90°
        if angle_deg > 90:
            angle_deg -= 180
        elif angle_deg < -90:
            angle_deg += 180

        target_quat = axisangle2quat(np.array([0, 0, 1]) * np.deg2rad(angle_deg))

        return {
            "pos": xpos + offset_xpos,
            "quat": quat_multiply(target_quat, null_quat),
            "grip": grip
        }


######################
######################
def example(
    num_episodes: int = 10,
    robots: str | list[str] = ["Panda"]*2,
    gripper_types: str | list[str] = ["default", "default"],
    num_recording_episodes: int = 0,
):
    env = suite.make(
        env_name="TwoArmLift",
        gripper_types=gripper_types,
        robots=robots,
        env_configuration="parallel",
        has_renderer=True,
        has_offscreen_renderer=num_recording_episodes > 0,
        use_camera_obs=num_recording_episodes > 0,
    )

    expert = TwoArmLiftWaypointExpert(
        environment=env,
        waypoints_file="two_arm_lift_wp.yaml",
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