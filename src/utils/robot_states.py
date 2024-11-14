from collections import OrderedDict

import numpy as np

class RawGripperState:
    """
    Container for the current gripper state, by storing the joint positions.
    """

    def __init__(
        self,
        qpos: np.array = np.zeros(2)
    ) -> None:
        """
        :param qpos: gripper joint positions
        """
        self._qpos = qpos

    @property
    def qpos(self) -> float:
        """
        :return: gripper state
        """
        return self._qpos

    @qpos.setter
    def qpos(self, qpos: np.array) -> None:
        """
        :param qpos: gripper joint positions
        """
        self._qpos = qpos

class EEState:
    """
    A class to represent the state of an end-effector.

    The quaternion is stored following the robosuite convention: [x, y, z, w]
    """
    def __init__(
        self,
        xyz: np.ndarray = np.zeros(3),
        quat: np.ndarray = np.array([1, 0, 0, 0]),
        grip: np.ndarray = np.zeros(2)
    ) -> None:
        """
        :param xyz: xyz position
        :param quat: quaternion [x, y, z, w]
        :param grip: gripper state
        """
        self._xyz = xyz
        self._quat = quat
        self._grip = RawGripperState(grip)

    @property
    def xyz(self) -> np.ndarray:
        """
        :return: xyz position
        """
        return self._xyz

    @property
    def quat(self) -> np.ndarray:
        """
        :return: quaternion [x, y, z, w]
        """
        return self._quat

    @property
    def qpos_grip(self) -> np.array:
        """
        :return: gripper state
        """
        return self._grip.qpos

    @property
    def gripper_state(self) -> RawGripperState:
        """
        :return: gripper state
        """
        return self._grip

    @xyz.setter
    def xyz(self, xyz: np.ndarray) -> None:
        """
        :param xyz: xyz position
        """
        self._xyz = xyz

    @quat.setter
    def quat(self, quat: np.ndarray) -> None:
        """
        :param quat: quaternion [x, y, z, w]
        """
        self._quat = quat

    @qpos_grip.setter
    def qpos_grip(self, qpos: np.array) -> None:
        """
        :param qpos: gripper state
        """
        self._grip.qpos = qpos

    @gripper_state.setter
    def gripper_state(self, grip: RawGripperState) -> None:
        """
        :param grip: gripper state
        """
        self._grip = grip

class TwoArmEEState:
    """
    A class to represent the state of two end-effectors.
    """
    def __init__(
        self,
        left: EEState = EEState(),
        right: EEState = EEState()
    ) -> None:
        """
        :param left: left end-effector state
        :param right: right end-effector state
        """
        self._left = left
        self._right = right

    @property
    def left(self) -> EEState:
        """
        :return: left end-effector state
        """
        return self._left

    @property
    def right(self) -> EEState:
        """
        :return: right end-effector state
        """
        return self._right

    @left.setter
    def left(self, left: EEState) -> None:
        """
        :param left: left end-effector state
        """
        self._left = left

    @right.setter
    def right(self, right: EEState) -> None:
        """
        :param right: right end-effector state
        """
        self._right = right

    @classmethod
    def from_dict(cls, data: OrderedDict, env_config: str = "parallel") -> "TwoArmGripperState":
        """
        Create a TwoArmGripperState from a dictionary.

        A TwoArmEnv of robosuite either has one or two robots.
        - One robot: environment config: "single-robot"
        - Two robots: environment config: "parallel" or "opposed"

        :param data: dictionary containing the left and right end-effector states
        :param env_config: environment configuration
        :return: TwoArmGripperState
        """
        if env_config == "single-robot":
            left = EEState(
                xyz=np.array(data["robot0_left_eef_pos"]),
                quat=np.array(data["robot0_left_eef_quat"]),
                grip=np.array(data["robot0_left_gripper_qpos"])
            )
            right = EEState(
                xyz=np.array(data["robot0_right_eef_pos"]),
                quat=np.array(data["robot0_right_eef_quat"]),
                grip=np.array(data["robot0_right_gripper_qpos"])
            )
        else:
            left = EEState(
                xyz=np.array(data["robot1_eef_pos"]),
                quat=np.array(data["robot1_eef_quat"]),
                grip=np.array(data["robot1_gripper_qpos"])
            )
            right = EEState(
                xyz=np.array(data["robot0_eef_pos"]),
                quat=np.array(data["robot0_eef_quat"]),
                grip=np.array(data["robot0_gripper_qpos"])
            )

        return cls(left, right)
