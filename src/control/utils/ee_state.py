import numpy as np
from transforms3d.euler import euler2quat, quat2euler
from transforms3d.quaternions import qmult

from src.control.utils.enums import GripperState

class EEState:
    """
    The EEState class holds a vector for
        - orientation (quaternion)
        - position (xyz)
        - gripper state (open or closed)

    ! currently no velocity or acceleration information is included

    NOTE: Quat is stored as w, x, y, z
    """

    def __init__(
            self,
            xyz: np.ndarray = np.zeros(3),
            rot : np.ndarray = np.zeros(3),
            grip: GripperState = GripperState.OPEN
    ):
        """
        :param xyz: xyz position
        :param rot: euler angles or quaternion
        :param grip: gripper state (open or closed)
        """

        assert len(xyz) == 3
        assert len(rot) == 3 or len(rot) == 4
        self.__xyz = np.array(xyz)
        self.__quat = np.array(rot) if len(rot) == 4 else np.array(euler2quat(*rot))

        self.__gripper_state = grip
        self.active = True

    def get_xyz(self) -> np.ndarray:
        """
        :return: xyz position
        """
        return self.__xyz

    def get_quat(self) -> np.ndarray:
        """
        :return: quaternion orientation
        """
        return self.__quat

    def get_abg(self) -> np.ndarray:
        """
        :return: euler angles
        """
        return np.asarray(quat2euler(self.__quat))

    def get_gripper_state(self) -> GripperState:
        """
        :return: gripper state
        """
        return self.__gripper_state

    def set_xyz(self, xyz: np.ndarray) -> None:
        """
        :param xyz: xyz position
        """
        assert len(xyz) == 3
        self.__xyz = np.asarray(xyz)

    def set_quat(self, quat: np.ndarray) -> None:
        """
        :param quat: quaternion orientation
        """
        assert len(quat) == 4
        self.__quat = np.asarray(quat)

    def update_quat(self, quat: np.ndarray) -> None:
        """
        :param quat: quaternion orientation
        """
        assert len(quat) == 4
        self.__quat += np.asarray(quat)

    def set_abg(self, abg: np.ndarray) -> None:
        """
        :param abg:
        """
        assert len(abg) == 3
        self.__quat = np.asarray(euler2quat(*abg))

    def set_gripper_state(self, gripper_state: GripperState) -> None:
        """
        :param gripper_state:
        """
        self.__gripper_state = gripper_state

    def set_all_quat(self, xyz: np.ndarray, quat: np.ndarray, grip : GripperState) -> None:
        """
        :param xyz:
        :param quat:
        :param grip:
        """
        assert len(xyz) == 3 and len(quat) == 4
        self.__xyz = np.asarray(xyz)
        self.__quat = np.asarray(quat)
        self.__gripper_state = grip

    def set_all_abg(self, xyz: np.ndarray, abg: np.ndarray, grip : GripperState) -> None:
        """
        :param xyz:
        :param abg:
        :param grip:
        """
        assert len(xyz) == 3 and len(abg) == 3
        self.__xyz = np.asarray(xyz)
        self.__quat = np.asarray(euler2quat(*abg))
        self.__gripper_state = grip

    @property
    def x(self) -> np.ndarray:
        """
        :return: x position
        """
        return self.__xyz[1]

    @x.setter
    def x(self, val: float) -> None:
        """
        :param val: x position
        """
        tmp = self.get_xyz()
        tmp[0] = val
        self.set_xyz(tmp)

    @property
    def y(self) -> np.ndarray:
        """
        :return: y position
        """
        return self.__xyz[1]

    @y.setter
    def y(self, val: float) -> None:
        """
        :param val: y position
        """
        tmp = self.get_xyz()
        tmp[1] = val
        self.set_xyz(tmp)

    @property
    def z(self) -> np.ndarray:
        """
        :return: z position
        """
        return self.__xyz[2]

    @z.setter
    def z(self, val: float) -> None:
        """
        :param val: z position
        """
        tmp = self.get_xyz()
        tmp[2] = val
        self.set_xyz(tmp)

    @property
    def gripper_state(self) -> GripperState:
        """
        :return: gripper state
        """
        return self.__gripper_state

    @gripper_state.setter
    def gripper_state(self, val: GripperState) -> None:
        """
        :param val: gripper state
        """
        self.__gripper_state = val

    def __str__(self):
        return f"xyz: {self.__xyz}, quat: {self.__quat}, grip: {self.__gripper_state}"

    def flatten(self) -> np.ndarray:
        return np.concatenate([self.__xyz, self.__quat, [self.__gripper_state]], axis=0)

    @classmethod
    def from_flattened(cls, array : np.ndarray):
        # euler angles or quaternion
        assert len(array) == 3 + 3 + 1 or len(array) == 3 + 4 + 1
        return cls(
            array[:3],
            array[3:-1],
            GripperState(array[-1])
        )



