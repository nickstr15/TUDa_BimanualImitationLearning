from typing import OrderedDict
import numpy as np

import robosuite as suite
from robosuite import TwoArmLift, TwoArmTransport
from robosuite.utils.transform_utils import quat2mat, mat2euler, axisangle2quat, quat_multiply

from src.demonstration.waypoints import TwoArmPickPlaceWaypointExpert
from src.utils.robot_targets import GripperTarget
from src.demonstration.waypoints.core.waypoint_expert import TwoArmWaypointExpertBase


class TwoArmTransportWaypointExpert(TwoArmPickPlaceWaypointExpert):
    """
    Specific waypoint expert for the TwoArmTransport environment.
    """
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


def example(
    n_episodes: int = 10,
    robots: str | list[str] = ["Panda"]*2,
    gripper_types: str | list[str] = ["default", "default"]
):
    two_arm_transport = suite.make(
        env_name="TwoArmTransport",
        robots=robots,
        gripper_types=gripper_types,
        env_configuration="parallel",
        tables_boundary=(0.65, 1.2, 0.05), # important for reachability
        bin_size=(0.3, 0.3, 0.08), # better reachability
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )

    expert = TwoArmTransportWaypointExpert(
        environment=two_arm_transport,
        waypoints_file="two_arm_transport_wp.yaml",
    )
    expert.visualize(n_episodes)


if __name__ == "__main__":
    example()
    #example(2, ["Kinova3", "Kinova3"])
    #example(2, ["IIWA", "IIWA"])
    #example(2, ["UR5e", "UR5e"], ["Robotiq140Gripper", "Robotiq140Gripper"])
    #example(2, ["Panda", "IIWA"])