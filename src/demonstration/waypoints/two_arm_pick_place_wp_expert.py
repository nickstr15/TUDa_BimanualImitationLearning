from typing import OrderedDict

import numpy as np
from transforms3d.euler import quat2euler
from transforms3d.quaternions import quat2mat, axangle2quat, qmult

from src.demonstration.waypoints.core.waypoint_expert import TwoArmWaypointExpertBase
from src.environments.manipulation.two_arm_pick_place import TwoArmPickPlace

import robosuite as suite

from src.utils.robot_states import TwoArmEEState, EEState
from src.utils.robot_targets import GripperTarget


class TwoArmPickPlaceWaypointExpert(TwoArmWaypointExpertBase):
    """
    Panda handover expert agent that follows a predefined trajectory of waypoints.

    Environment must have use_object_obs=True.
    """

    def __init__(
            self,
            env : TwoArmPickPlace,
        ):
        """
        Constructor for the PandaHandoverWpExpertBase class.
        :param env: environment to interact with
        """
        super().__init__(
            environment=env,
            waypoints_file="two_arm_pick_place_wp.yaml",
        )

    ######################################################################
    # Definition of special ee targets ###################################
    ######################################################################
    def _create_ee_target_methods_dict(self) -> dict:
        """
        Create a dictionary of methods to get the special end-effector targets.

        All methods take the observation after reset as input.

        :return: dictionary with the methods
        """
        return {
            "initial_state_left": self.__initial_state_left,
            "initial_state_right": self.__initial_state_right,

            "pre_pick_up_robot_left": self.__pre_pick_up_robot_left,
            "pre_drop_off_robot_right": self.__pre_drop_off_robot_right,
        }

    def __initial_state_left(self, obs: OrderedDict = None) -> dict:
        """
        Initial state for the left arm.

        :param obs: observation after reset
        :return: dictionary with the target position, orientation, and gripper state
        """
        state: EEState = TwoArmEEState.from_dict(obs, env_config=self._env.env_configuration).left
        dct = {
            "pos": state.xyz + np.array([0.0, 0.0, 0.05]), #TODO remove shift after testing
            "quat": state.quat,
            "grip": GripperTarget.OPEN_VALUE
        }
        return dct

    def __initial_state_right(self, obs: OrderedDict = None) -> dict:
        """
        Initial state for the right arm.

        :param obs: observation after reset
        :return: dictionary with the target position, orientation, and gripper state
        """
        state: EEState = TwoArmEEState.from_dict(obs, env_config=self._env.env_configuration).right
        dtc = {
            "pos": state.xyz + np.array([0.0, 0.0, -0.1]), #TODO remove shift after testing
            "quat": state.quat,
            "grip": GripperTarget.CLOSED_VALUE
        }
        return dtc

    def __pre_pick_up_robot_left(self, obs: OrderedDict = None) -> dict:
        """
        Pre-pick up position for the second arm.

        :param obs: observation after reset
        :return: dictionary with the target position, orientation, and gripper state
        """
        raise NotImplementedError

    def __pre_drop_off_robot_right(self, obs: OrderedDict = None) -> dict:
        """
        Pre-drop off position for the right arm.

        :param obs: observation after reset
        :return: dictionary with the target position, orientation, and gripper state
        """
        raise NotImplementedError

    def __get_target(self, name: str, offset: np.ndarray, grip : float) -> dict:
        """
        Get the target position and orientation for the end-effector.
        :param name: name of the body
        :param offset: offset from the body
        :param grip: gripper state

        :return: dictionary with the target position, orientation, and gripper state
        """
        return {
            "pos": np.array([0.0, 0.0, 0.0]),
            "quat": np.array([1.0, 0.0, 0.0, 0.0]),
            "grip": grip
        }

        # TODO: Implement the following code snippet
        """
        quat, pos = ..., ...

        # the objects are only rotated around the z-axis
        _, _, angle = np.rad2deg(quat2euler(quat))
        # normalize the angle to the range -180° to 180°
        angle = (angle + 180) % 360 - 180
        # map it to its symmetrical equivalent in the range -90° to 90°
        angle = angle - 180 if angle > 90 else (
            angle + 180 if angle < -90 else angle)
        # back to radians
        angle = np.deg2rad(angle)
        quat = axangle2quat([0, 0, 1], angle)

        ee_quat = qmult(quat, self._zero_quat_ee)
        ee_pos = pos + np.dot(quat2mat(quat), offset)

        return {
            "pos": ee_pos,
            "quat": ee_quat,
            "grip": grip
        }
        """

if __name__ == "__main__":
    two_arm_pick_place = suite.make(
        env_name="TwoArmPickPlace",
        robots=["Baxter"],
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )

    expert = TwoArmPickPlaceWaypointExpert(two_arm_pick_place)
    expert.visualize(10)
    expert.dispose()