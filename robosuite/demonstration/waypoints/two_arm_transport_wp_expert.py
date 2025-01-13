from typing import OrderedDict
import numpy as np

import robosuite as suite
from robosuite import TwoArmLift, TwoArmTransport
from robosuite.utils.transform_utils import quat2mat, mat2euler, axisangle2quat, quat_multiply

from robosuite.demonstration.waypoints import TwoArmPickPlaceWaypointExpert
from robosuite.utils.robot_targets import GripperTarget
from robosuite.demonstration.waypoints.core.waypoint_expert import TwoArmWaypointExpertBase


class TwoArmTransportWaypointExpert(TwoArmPickPlaceWaypointExpert):
    """
    Specific waypoint expert for the TwoArmTransport environment.
    """
    target_env_name = "TwoArmTransport"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._env: TwoArmTransport = self._env  # for type hinting in pycharm

        self._hammer_name = "payload"
        self._get_hammer_handle_length_fn = lambda: self._env.transport.payload.handle_length
        self._get_hammer_head_halfsize_fn = lambda: self._env.transport.payload.head_halfsize
        self._bin_height = self._env.bin_size[2]
        self._target_bin_name = "target_bin"


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
            "pre_lid_handle_right": self.__pre_lid_handle_right,
            "pre_trash_left": self.__pre_trash_left,

            "lid_handle_right": self.__lid_handle_right,
            "trash_left": self.__trash_left,

            "pre_lid_drop_right": self.__pre_lid_drop_right,
            "pre_trash_drop_left": self.__pre_trash_drop_left,

            "lid_drop_right": self.__lid_drop_right,
        }

        dct.update(update)
        return dct

    def __pre_lid_handle_right(self, obs: OrderedDict = None) -> dict:
        """
        Pre-lid handle position for the right arm.

        :param obs: observation after reset
        :return: dictionary with the target position, orientation, and gripper state
        """
        dct = self._calculate_target_pose(
            obj_pos=obs["lid_handle_pos"],
            obj_quat=obs["lid_handle_quat"],
            offset=np.array([0.0, 0.0, 0.1]),
            gripper_state=GripperTarget.OPEN_VALUE,
        )

        return dct

    def __pre_trash_left(self, obs: OrderedDict = None) -> dict:
        """
        Pre-trash position for the left arm.

        :param obs: observation after reset
        :return: dictionary with the target position, orientation, and gripper state
        """
        dct = self._calculate_target_pose(
            obj_pos=obs["trash_pos"],
            obj_quat=obs["trash_quat"],
            offset=np.array([0.0, 0.0, self._env.bin_size[2] + 0.05]),
            gripper_state=GripperTarget.OPEN_VALUE,
        )

        return dct

    def __lid_handle_right(self, obs: OrderedDict = None) -> dict:
        """
        Lid handle position for the right arm.

        :param obs: observation after reset
        :return: dictionary with the target position, orientation, and gripper state
        """
        dct = self._calculate_target_pose(
            obj_pos=obs["lid_handle_pos"],
            obj_quat=obs["lid_handle_quat"],
            offset=np.array([0.0, 0.0, 0.0]),
            gripper_state=GripperTarget.OPEN_VALUE,
        )

        return dct

    def __trash_left(self, obs: OrderedDict = None) -> dict:
        """
        Trash position for the left arm.

        :param obs: observation after reset
        :return: dictionary with the target position, orientation, and gripper state
        """
        dct = self._calculate_target_pose(
            obj_pos=obs["trash_pos"],
            obj_quat=obs["trash_quat"],
            offset=np.array([0.0, 0.0, 0.0]),
            gripper_state=GripperTarget.OPEN_VALUE,
        )

        return dct

    def __pre_lid_drop_right(self, obs: OrderedDict = None) -> dict:
        """
        Pre-lid drop position for the right arm.

        :param obs: observation after reset
        :return: dictionary with the target position, orientation, and gripper state
        """
        lid_handle_offset = np.array([self._env.table_full_size[0] / 2, 0.0, 0.0])
        dct = self._calculate_target_pose(
            obj_pos=obs["lid_handle_pos"] - lid_handle_offset,
            obj_quat=obs["lid_handle_quat"],
            offset=np.array([0.0, 0.0, 0.1]),
            gripper_state=GripperTarget.CLOSED_VALUE,
        )

        return dct

    def __pre_trash_drop_left(self, obs: OrderedDict = None) -> dict:
        """
        Pre-trash drop position for the left arm.

        :param obs: observation after reset
        :return: dictionary with the target position, orientation, and gripper state
        """
        dct = self._calculate_target_pose(
            obj_pos=obs["trash_bin_pos"],
            obj_quat=obs["trash_quat"],
            offset=np.array([0.0, 0.0, self._env.bin_size[2] + 0.05]),
            gripper_state=GripperTarget.CLOSED_VALUE,
        )

        return dct

    def __lid_drop_right(self, obs: OrderedDict = None) -> dict:
        """
        Lid drop position for the right arm.

        :param obs: observation after reset
        :return: dictionary with the target position, orientation, and gripper state
        """
        target_pos = obs["lid_handle_pos"] - np.array([self._env.table_full_size[0] / 2, 0.0, 0.0])
        target_pos[2] = self._env.table_offsets[0][2] + 0.05
        dct = self._calculate_target_pose(
            obj_pos=target_pos,
            obj_quat=obs["lid_handle_quat"],
            offset=np.array([0.0, 0.0, 0.0]),
            gripper_state=GripperTarget.CLOSED_VALUE,
        )

        return dct

if __name__ == "__main__":
    expert = TwoArmTransportWaypointExpert
    f = "two_arm_transport_wp.yaml"

    expert.example(f)
    #expert.example(f, robots=["Kinova3"]*2)
    #expert.example(f, robots=["IIWA"]*2)
    #expert.example(f, robots=["UR5e"]*2, gripper_types=["Robotiq85Gripper"]*2)
    # expert.example(f, robots=["Panda", "IIWA"])