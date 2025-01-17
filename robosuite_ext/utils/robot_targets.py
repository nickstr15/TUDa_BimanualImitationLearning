import numpy as np
from robosuite.utils.transform_utils import quat2axisangle, quat_multiply, quat_inverse

from robosuite_ext.utils.robot_states import TwoArmEEState, EEState


class GripperTarget:
    """
    Container for a gripper target.

    Following the robosuite convention:
    -1 => open, 1 => closed
    """
    OPEN_VALUE = -1.0
    CLOSED_VALUE = 1.0

    def __init__(
        self,
        grip: float = OPEN_VALUE
    ) -> None:
        """
        :param grip: gripper value
        """
        self._grip = np.clip(
            grip,
            min(self.OPEN_VALUE, self.CLOSED_VALUE),
            max(self.OPEN_VALUE, self.CLOSED_VALUE)
        )

    @property
    def grip(self) -> float:
        """
        :return: gripper value
        """
        return self._grip

    @grip.setter
    def grip(self, grip: float) -> None:
        """
        :param grip: gripper value
        """
        self._grip = np.clip(
            grip,
            min(self.OPEN_VALUE, self.CLOSED_VALUE),
            max(self.OPEN_VALUE, self.CLOSED_VALUE)
        )


class EETarget:
    """
    A class to represent a target for an end-effector.

    The quaternion is stored following the robosuite convention: [x, y, z, w]
    """
    def __init__(
        self,
        xyz: np.ndarray = np.zeros(3),
        quat: np.ndarray = np.array([1, 0, 0, 0]),
        grip: float = GripperTarget.OPEN_VALUE,
        pos_tol: float = 0.01,
        ori_tol: float = np.deg2rad(5),
        max_vel_pos: float = None,
        max_vel_ori: float = None,
    ) -> None:
        """
        :param xyz: xyz position
        :param quat: quaternion [x, y, z, w]
        :param grip: gripper state
        :param pos_tol: position tolerance
        :param ori_tol: orientation tolerance
        """
        self._xyz = xyz
        self._quat = quat
        self._grip = GripperTarget(grip)
        self._pos_tol = pos_tol
        self._ori_tol = ori_tol
        self._max_vel_pos = max_vel_pos
        self._max_vel_ori = max_vel_ori

    @property
    def xyz(self) -> np.ndarray:
        """
        :return: xyz position
        """
        return self._xyz

    @xyz.setter
    def xyz(self, xyz: np.ndarray) -> None:
        """
        :param xyz: xyz position
        """
        self._xyz = xyz

    @property
    def quat(self) -> np.ndarray:
        """
        :return: quaternion [x, y, z, w]
        """
        return self._quat

    @quat.setter
    def quat(self, quat: np.ndarray) -> None:
        """
        :param quat: quaternion [x, y, z, w]
        """
        self._quat = quat

    @property
    def grip(self) -> float:
        """
        :return: gripper value
        """
        return self._grip.grip

    @grip.setter
    def grip(self, grip: float) -> None:
        """
        :param grip: gripper value
        """
        self._grip.grip = grip

    @property
    def gripper_state(self) -> GripperTarget:
        """
        :return: gripper state
        """
        return self._grip

    @gripper_state.setter
    def gripper_state(self, gripper_state: GripperTarget) -> None:
        """
        :param gripper_state: gripper state
        """
        self._grip = gripper_state

    @property
    def pos_tol(self) -> float:
        """
        :return: position tolerance
        """
        return self._pos_tol

    @pos_tol.setter
    def pos_tol(self, pos_tol: float) -> None:
        """
        :param pos_tol: position tolerance
        """
        self._pos_tol = pos_tol

    @property
    def ori_tol(self) -> float:
        """
        :return: orientation tolerance
        """
        return self._ori_tol

    @ori_tol.setter
    def ori_tol(self, ori_tol: float) -> None:
        """
        :param ori_tol: orientation tolerance
        """
        self._ori_tol = ori_tol

    @property
    def max_vel_pos(self) -> float | None:
        """
        :return: maximum velocity for position
        """
        return self._max_vel_pos

    @max_vel_pos.setter
    def max_vel_pos(self, max_vel_pos: float | None) -> None:
        """
        :param max_vel_pos: maximum velocity for position
        """
        self._max_vel_pos = max_vel_pos

    @property
    def max_vel_ori(self) -> float | None:
        """
        :return: maximum velocity for orientation
        """
        return self._max_vel_ori

    @max_vel_ori.setter
    def max_vel_ori(self, max_vel_ori: float | None) -> None:
        """
        :param max_vel_ori: maximum velocity for orientation
        """
        self._max_vel_ori = max_vel_ori

    def is_reached_by(self, current_state: EEState) -> bool:
        """
        Check if the target state is reached by the current state.
        :param current_state: Current state
        :return: True if the target state is reached by the current state
        """
        pos_diff = np.linalg.norm(self._xyz - current_state.xyz)
        ori_diff = np.linalg.norm(
            quat2axisangle(quat_multiply(quat_inverse(self._quat), current_state.quat))[1:]
        )
        return pos_diff < self._pos_tol and ori_diff < self._ori_tol

class TwoArmEETarget:
    """
    A class to represent a target for two end-effectors.
    """
    def __init__(
        self,
        left: EETarget = EETarget(),
        right: EETarget = EETarget()
    ) -> None:
        """
        :param left: left end-effector state
        :param right: right end-effector state
        """
        self._left = left
        self._right = right

    @property
    def left(self) -> EETarget:
        """
        :return: left end-effector state
        """
        return self._left

    @property
    def right(self) -> EETarget:
        """
        :return: right end-effector state
        """
        return self._right

    @left.setter
    def left(self, left: EETarget) -> None:
        """
        :param left: left end-effector state
        """
        self._left = left

    @right.setter
    def right(self, right: EETarget) -> None:
        """
        :param right: right end-effector state
        """
        self._right = right

    def is_reached_by(self, current_state: TwoArmEEState) -> bool:
        """
        Check if the target state is reached by the current state.
        :param current_state: Current state
        :return: True if the target state is reached by the current state
        """

        return self._left.is_reached_by(current_state.left) and self._right.is_reached_by(current_state.right)
