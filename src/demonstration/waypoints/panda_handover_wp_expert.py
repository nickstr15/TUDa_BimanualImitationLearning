import os

import numpy as np
from transforms3d.axangles import axangle2mat
from transforms3d.euler import quat2euler, euler2mat, euler2quat
from transforms3d.quaternions import quat2mat, mat2quat, axangle2quat, qmult

from src.control.utils.enums import GripperState
from src.demonstration.waypoints.core.waypoint import Waypoint
from src.demonstration.waypoints.core.waypoint_expert import WaypointExpertBase
from src.environments import PandaHandoverEnv
from src.environments.core.enums import ActionMode


class PandaHandoverWpExpert(WaypointExpertBase):
    """
    Panda handover expert agent that follows a predefined trajectory of waypoints.
    """

    def __init__(
            self,
            dual_panda_env_args : dict = None,
            **kwargs):
        """
        Constructor for the PandaHandoverWpExpertBase class.
        :param dual_panda_env_args: additional arguments for the environment, default is None
        :param kwargs: additional arguments for the expert, see WaypointExpert (core.waypoint_expert)
        """
        super().__init__(
            environment=PandaHandoverEnv(**dual_panda_env_args if dual_panda_env_args else {}),
            waypoints_file="panda_handover_wp.yaml",
            **kwargs
        )

        # for type hinting
        self._env : PandaHandoverEnv = self._env


    ######################################################################
    # Definition of special ee targets ###################################
    ######################################################################

    # rotation of the end-effector in the world frame that aligns with cuboid rotation = 0°
    _zero_quat_ee = np.array([0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0])

    def _create_ee_target_methods_dict(self) -> dict:
        """
        Create a dictionary of methods to get the special end-effector targets.
        """
        return {
            "home_panda_01": self.__home_panda_01,
            "home_panda_02": self.__home_panda_02,

            "pre_pick_up_panda_02": self.__pre_pick_up_panda_02,

            "pre_drop_off_panda_01": self.__pre_drop_off_panda_01,
        }

    def __home_panda_01(self) -> dict:
        """
        Home for the first Panda robot.
        """
        home_target = self._env.x_home_targets["panda_01"]
        return {
            "pos": home_target.get_xyz(),
            "quat": home_target.get_quat(),
            "grip": home_target.get_gripper_state()
        }

    def __home_panda_02(self) -> dict:
        """
        Home for the second Panda robot.
        """
        home_target = self._env.x_home_targets["panda_02"]
        return {
            "pos": home_target.get_xyz(),
            "quat": home_target.get_quat(),
            "grip": home_target.get_gripper_state()
        }

    def __pre_pick_up_panda_02(self) -> dict:
        """
        Pre-pick up position for the second Panda robot.
        """
        return self.__get_target("cuboid_center", np.array([0.0, 0.03, 0.05]), GripperState.OPEN)

    def __pre_drop_off_panda_01(self) -> dict:
        """
        Pre-drop off position for the first Panda robot.
        """
        return self.__get_target("box_center", np.array([0.0, 0.03, 0.1]), GripperState.CLOSED)

    def __get_target(self, body_name: str, offset: np.ndarray, grip : GripperState) -> dict:
        """
        Get the target position and orientation for the end-effector.
        :param body_name: name of the body
        :param offset: offset from the body
        :param grip: gripper state

        :return: dictionary with the target position, orientation, and gripper state
        """
        cuboid_quat, cuboid_pos = self._env._get_object_quat_pos()[body_name]

        # the cuboid is only rotated around the z-axis
        _, _, cuboid_angle = np.rad2deg(quat2euler(cuboid_quat))
        # normalize the angle to the range -180° to 180°
        cuboid_angle = (cuboid_angle + 180) % 360 - 180
        # map it to its symmetrical equivalent in the range -90° to 90°
        cuboid_angle = cuboid_angle - 180 if cuboid_angle > 90 else (
            cuboid_angle + 180 if cuboid_angle < -90 else cuboid_angle)
        # back to radians
        cuboid_angle = np.deg2rad(cuboid_angle)
        cuboid_quat = axangle2quat([0, 0, 1], cuboid_angle)

        ee_quat = qmult(cuboid_quat, self._zero_quat_ee)
        ee_pos = cuboid_pos + np.dot(quat2mat(cuboid_quat), offset)

        return {
            "pos": ee_pos,
            "quat": ee_quat,
            "grip": grip
        }

if __name__ == "__main__":
    env_args = dict(
        visualize_targets=True,
        action_mode = ActionMode.RELATIVE
    )

    expert = PandaHandoverWpExpert(env_args)
    expert.visualize(10)
    expert.dispose()