from typing import OrderedDict, Callable
import numpy as np

import robosuite as suite
from robosuite import TwoArmHandover
from robosuite.utils.transform_utils import quat2mat, mat2euler, axisangle2quat, quat_multiply

from src.utils.robot_targets import GripperTarget
from src.demonstration.waypoints.core.waypoint_expert import TwoArmWaypointExpertBase


class TwoArmHandoverWaypointExpert(TwoArmWaypointExpertBase):
    """
    Specific waypoint expert for the TwoArmHandover environment.
    """
    target_env_name = "TwoArmHandover"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._env: TwoArmHandover = self._env  # for type hinting in pycharm

        self._handover_mode = 1 # track on which side to position arm1 for handover, 1 => behind arm0, -1 => in front of arm0
        self._hammer_name = "hammer"
        self._get_hammer_handle_length_fn = lambda: self._env.hammer.handle_length
        self._bin_height = 0

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
            "pre_pick_up_hammer_right": self.__pre_pick_up_hammer_right,
            "pick_up_hammer_right": self.__pick_up_hammer_right,

            "hand_over_robot_right": self.__hand_over_robot_right,
            "pre_hand_over_robot_left": self.__pre_hand_over_robot_left,
        }

        dct.update(update)
        return dct

    def __pre_pick_up_hammer_right(self, obs: OrderedDict = None) -> dict:
        """
        Pre-pick up position for the right arm.

        :param obs: observation after reset
        :return: dictionary with the target position, orientation, and gripper state
        """
        hammer_quat = obs[f"{self._hammer_name}_quat"]
        flip_sign = np.sign(mat2euler(quat2mat(hammer_quat))[0])

        handle_length = self._get_hammer_handle_length_fn()
        grip_offset = 0.5 * handle_length - 0.02  # grasp 2cm from hammer head
        grip_offset *= -1*flip_sign
        self._handover_mode = -1*flip_sign

        dct, flip_sign = self._calculate_target_pose_with_flip_sign(
            obj_pos=obs[f"{self._hammer_name}_pos"],
            obj_quat=obs[f"{self._hammer_name}_quat"],
            offset=np.array([grip_offset, 0.0, self._bin_height+0.1]),
            gripper_state=GripperTarget.OPEN_VALUE
        )
        self._handover_mode *= flip_sign
        return dct
    
    def __pick_up_hammer_right(self, obs: OrderedDict = None) -> dict:
        """
        Pick up position for the right arm.

        :param obs: observation after reset
        :return: dictionary with the target position, orientation, and gripper state
        """
        hammer_quat = obs[f"{self._hammer_name}_quat"]
        flip_sign = np.sign(mat2euler(quat2mat(hammer_quat))[0])

        handle_length = self._get_hammer_handle_length_fn()
        grip_offset = 0.5 * handle_length - 0.02  # grasp 2cm from hammer head
        grip_offset *= -1 * flip_sign

        dct = self._calculate_target_pose(
            obj_pos=obs[f"{self._hammer_name}_pos"],
            obj_quat=obs[f"{self._hammer_name}_quat"],
            offset=np.array([grip_offset, 0.0, 0.0]),
            gripper_state=GripperTarget.OPEN_VALUE
        )
        return dct

    def _calculate_target_pose(
            self,
            obj_pos: np.ndarray,
            obj_quat: np.ndarray,
            offset: np.ndarray = np.zeros(3),
            gripper_state: float = GripperTarget.OPEN_VALUE,
            adjustment_quat: np.array = np.array([0, 0, 0, 1.0])) -> dict:
        """
        Calculates the target position, orientation, and gripper state based on the object's position and quaternion.

        :param obj_pos: Position of the object (e.g., hammer or bin)
        :param obj_quat: Quaternion orientation of the object
        :param offset: Offset to apply to the position
        :param gripper_state: Desired gripper state (e.g., open or closed)
        :param adjustment_quat: Additional angle adjustment for the orientation
        :return: Dictionary with the target position, orientation, and gripper state
        """
        return self._calculate_target_pose_with_flip_sign(obj_pos, obj_quat, offset, gripper_state, adjustment_quat)[0]

    def _calculate_target_pose_with_flip_sign(
            self,
            obj_pos: np.ndarray,
            obj_quat: np.ndarray,
            offset: np.ndarray = np.zeros(3),
            gripper_state: float = GripperTarget.OPEN_VALUE,
            adjustment_quat: np.array = np.array([0, 0, 0, 1.0])) -> tuple[dict, int]:
        """
        Calculates the target position, orientation, and gripper state based on the object's position and quaternion.

        :param obj_pos: Position of the object (e.g., hammer or bin)
        :param obj_quat: Quaternion orientation of the object
        :param offset: Offset to apply to the position
        :param gripper_state: Desired gripper state (e.g., open or closed)
        :param adjustment_quat: Additional angle adjustment for the orientation
        :return: Dictionary with the target position, orientation, and gripper state + flip sign
        """
        # Calculate angle in degrees and map to symmetrical range
        angle_deg = np.rad2deg(mat2euler(quat2mat(obj_quat))[2]) + 90

        # Map to range [-180, 180]
        angle_deg = (angle_deg + 180) % 360 - 180

        # Map angle to range [-90, 90], due to symmetry and adjust offset if necessary
        flip_sign = 1
        if angle_deg > 90:
            offset[0] *= -1
            angle_deg -= 180
            flip_sign *= -1
        elif angle_deg < -90:
            offset[0] *= -1
            angle_deg += 180
            flip_sign *= -1

        # Apply angle adjustment and convert to quaternion
        angle_rad = np.deg2rad(angle_deg)
        target_quat = axisangle2quat(np.array([0, 0, 1]) * angle_rad)
        target_quat = quat_multiply(target_quat, self._null_quat_left)

        #apply additional angle adjustment
        target_quat = quat_multiply(adjustment_quat, target_quat)

        # FIXME: invert z offset for correct behavior
        offset[2] *= -1

        # Calculate target position with offset
        target_pos = obj_pos + np.dot(quat2mat(target_quat), offset)

        dct = {
            "pos": target_pos,
            "quat": target_quat,
            "grip": gripper_state
        }
        return dct, flip_sign

    def __hand_over_robot_right(self, obs: OrderedDict = None) -> dict:
        """
        Hand-over position for the right arm.

        :param obs: observation after reset
        :return: dictionary with the target position, orientation, and gripper state
        """
        target_pos = np.array([-0.1, 0.0, 1])
        target_quat = quat_multiply(
            axisangle2quat(np.array([1, 0, 0]) * np.deg2rad(90)),
            self._null_quat_right
        )
        return {
            "pos": target_pos,
            "quat": target_quat,
            "grip": GripperTarget.CLOSED_VALUE
        }

    def __pre_hand_over_robot_left(self, obs: OrderedDict = None) -> dict:
        target_pos = np.array([-0.1, 0, 1.1])
        target_pos[0] += -0.06*self._handover_mode
        target_quat = self._null_quat_right
        return {
            "pos": target_pos,
            "quat": target_quat,
            "grip": GripperTarget.OPEN_VALUE
        }

if __name__ == "__main__":
    expert = TwoArmHandoverWaypointExpert
    f = "two_arm_handover_wp.yaml"

    expert.example(f)
    #expert.example(f, robots=["Kinova3"]*2)
    #expert.example(f, robots=["IIWA"]*2)
    #expert.example(f, robots=["UR5e"]*2, gripper_types=["Robotiq85Gripper"]*2)
    # expert.example(f, robots=["Panda", "IIWA"])