import numpy as np
from transforms3d.euler import euler2quat, quat2euler

from src.control.utils.enums import GripperState

class ArmState:
    """
    The ArmState class holds a vector for
        - orientation (quaternion)
        - position (xyz)
        - gripper state (open or closed)

    NOTE: Quat is stored as w, x, y, z
    """

    def __init__(
            self,
            xyz_rot: np.ndarray = np.zeros(6),
            xyz_rot_vel: np.ndarray = np.zeros(6),
            grip: GripperState = GripperState.OPEN
    ):
        """
        :param xyz_rot: xyz position and euler angles
        :param xyz_rot_vel: xyz velocity and euler angles velocity o
        :param grip: gripper state (open or closed)
        """

        assert (len(xyz_rot) == 6 or len(xyz_rot) == 7) and (len(xyz_rot_vel) == 6 or len(xyz_rot_vel) == 7)
        assert np.all([isinstance(x, float) or isinstance(x, int) for x in xyz_rot])
        self.__xyz = np.array(xyz_rot)[:3]
        self.__xyz_vel = np.array(xyz_rot_vel)[:3]
        self.__quat = np.array(euler2quat(*xyz_rot[3:])) if len(xyz_rot) == 6 else np.array(xyz_rot[3:])
        self.__quat_vel = np.array(euler2quat(*xyz_rot_vel[3:])) if len(xyz_rot_vel) == 6 else np.array(xyz_rot_vel[3:])

        self.__gripper_state = grip
        self.active = True

    def get_xyz(self) -> np.ndarray:
        """
        :return: xyz position
        """
        return self.__xyz

    def get_xyz_vel(self) -> np.ndarray:
        """
        :return: xyz velocity
        """
        return self.__xyz_vel

    def get_quat(self) -> np.ndarray:
        """
        :return: quaternion orientation
        """
        return self.__quat

    def get_quat_vel(self) -> np.ndarray:
        """
        :return: quaternion velocity
        """
        return np.asarray(self.__quat_vel)

    def get_abg(self) -> np.ndarray:
        """
        :return: euler angles
        """
        return np.asarray(quat2euler(self.__quat))

    def get_abg_vel(self) -> np.ndarray:
        """
        :return: euler angles
        """
        return np.asarray(quat2euler(self.__quat_vel))

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

    def set_xyz_vel(self, xyz_vel: np.ndarray) -> None:
        """
        :param xyz_vel: xyz velocity
        """
        assert len(xyz_vel) == 3
        self.__xyz_vel = np.asarray(xyz_vel)


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

    def set_quat_vel(self, quat_vel: np.ndarray) -> None:
        """
        :param quat_vel:
        """
        assert len(quat_vel) == 4
        self.__quat_vel = np.asarray(quat_vel)

    def set_abg(self, abg: np.ndarray) -> None:
        """
        :param abg:
        """
        assert len(abg) == 3
        self.__quat = np.asarray(euler2quat(*abg))

    def set_abg_vel(self, abg_vel: np.ndarray) -> None:
        """
        :param abg_vel:
        """
        assert len(abg_vel) == 3
        self.__quat_vel = np.asarray(euler2quat(*abg_vel))

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



