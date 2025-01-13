from typing import OrderedDict
import numpy as np

import robosuite as suite
from robosuite.utils.transform_utils import quat2mat, mat2euler, axisangle2quat, quat_multiply, quat2axisangle

from robosuite.demonstration.waypoints.two_arm_handover_wp_expert import TwoArmHandoverWaypointExpert
from robosuite.environments import TwoArmPickPlace
from robosuite.environments.manipulation.two_arm_hinged_bin import TwoArmHingedBin
from robosuite.utils.robot_targets import GripperTarget


class TwoArmHingedBinWaypointExpert(TwoArmHandoverWaypointExpert):
    """
    Specific waypoint expert for the TwoArmPickPlace environment.
    """
    target_env_name = "TwoArmHingedBin"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._env : TwoArmHingedBin = self._env # for type hinting in pycharm

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
            "pre_handle_left": self.__pre_handle_left,
            "handle_left": self.__handle_left,

            "open_bin_left": self.__open_bin_left,

            "pre_drop_off_hammer_right": self.__pre_drop_off_hammer_right,
            "drop_off_hammer_right": self.__drop_off_hammer_right,
        }

        dct.update(update)
        return dct

    def __pre_handle_left(self, obs: OrderedDict = None) -> dict:
        """
        Pre-lid handle position for the left arm.

        :param obs: observation after reset
        :return: dictionary with the target position, orientation, and gripper state
        """
        dct = self._calculate_target_pose(
            obj_pos=obs["handle_pos"],
            obj_quat=obs["handle_quat"],
            offset=np.array([0.0, 0.0, 0.1]),
            gripper_state=GripperTarget.OPEN_VALUE,
        )

        return dct

    def __handle_left(self, obs: OrderedDict = None) -> dict:
        """
        Lid handle position for the right arm.

        :param obs: observation after reset
        :return: dictionary with the target position, orientation, and gripper state
        """
        dct = self._calculate_target_pose(
            obj_pos=obs["handle_pos"],
            obj_quat=obs["handle_quat"],
            offset=np.array([0.0, 0.0, 0.0]),
            gripper_state=GripperTarget.OPEN_VALUE,
        )

        return dct

    def __open_bin_left(self, obs: OrderedDict = None) -> dict:
        """
        Position for the left ee to open the bin
        :param obs: observation after reset
        :return: dictionary with the target position, orientation, and gripper state
        """
        hinge_pos = obs["hinge_pos"]
        handle_pos = obs["handle_pos"]


        # vector from hinge to handle
        hinge_handle = handle_pos - hinge_pos
        unit_z = np.array([0, 0, 1])
        rotation_direction = np.cross(unit_z, hinge_handle)
        rotation_direction /= np.linalg.norm(rotation_direction)

        rotation_axisangle = rotation_direction * np.deg2rad(-80)
        rotation_quat = axisangle2quat(rotation_axisangle)
        rotation_mat = quat2mat(rotation_quat)

        hinge_handle_open = np.dot(rotation_mat, hinge_handle)
        handle_open = hinge_pos + hinge_handle_open

        dct = self.__handle_left(obs)
        dct["pos"] = handle_open
        dct["quat"] = quat_multiply(rotation_quat, dct["quat"])
        dct["grip"] = GripperTarget.CLOSED_VALUE
        return dct


    def __pre_drop_off_hammer_right(self, obs: OrderedDict = None) -> dict:
        """
        Pre-drop off position for the left arm.

        :param obs: observation after reset
        :return: dictionary with the target position, orientation, and gripper state
        """
        return self.__drop_off_computation(
            obs=obs,
            z_offset=self._bin_height + 0.2
        )

    def __drop_off_hammer_right(self, obs: OrderedDict = None) -> dict:
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
        dct = self._calculate_target_pose(
            obj_pos=obs[f"{self._target_bin_name}_pos"],
            obj_quat=obs.get(f"{self._target_bin_name}_quat", np.array([0, 0, 0, 1])),
            offset=np.array([0.0, 0.0, z_offset]),
            gripper_state=GripperTarget.CLOSED_VALUE,
            adjustment_quat=axisangle2quat(np.array([0, 0, 1]) * np.pi/4)
        )
        return dct

if __name__ == "__main__":
    expert = TwoArmHingedBinWaypointExpert
    f = "two_arm_hinged_bin_wp.yaml"

    expert.example(f)
    #expert.example(f, robots=["Kinova3"]*2)
    #expert.example(f, robots=["IIWA"]*2)
    #expert.example(f, robots=["UR5e"]*2, gripper_types=["Robotiq85Gripper"]*2)
    #expert.example(f, robots=["Panda", "IIWA"])