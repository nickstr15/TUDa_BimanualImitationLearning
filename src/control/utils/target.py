from enum import Enum

import numpy as np
from transforms3d.euler import euler2quat, quat2euler

class GripperState(Enum):
    OPEN = 255
    CLOSED = 0

class SingleArmTarget:
    """
    The Target class holds a target vector for
        - orientation (quaternion)
        - position (xyz)
        - gripper state (open or closed)

    NOTE: Quat is stored as w, x, y, z
    """

    def __init__(
            self,
            xyz_abg: np.ndarray = np.zeros(6),
            xyz_abg_vel: np.ndarray = np.zeros(6),
            grip: GripperState = GripperState.OPEN
    ):
        """
        :param xyz_abg: xyz position and euler angles of the target
        :param xyz_abg_vel: xyz velocity and euler angles velocity of the target
        :param grip: gripper state of the target
        """

        assert len(xyz_abg) == 6 and len(xyz_abg_vel) == 6
        assert np.all([isinstance(x, float) or isinstance(x, int) for x in xyz_abg])
        self.__xyz = np.array(xyz_abg)[:3]
        self.__xyz_vel = np.array(xyz_abg_vel)[:3]
        self.__quat = np.array(euler2quat(*xyz_abg[3:]))
        self.__quat_vel = np.array(euler2quat(*xyz_abg_vel[3:]))

        self.__gripper_state = grip
        self.active = True

    def get_xyz(self) -> np.ndarray:
        """
        :return: xyz position of the target
        """
        return self.__xyz

    def get_xyz_vel(self) -> np.ndarray:
        """
        :return: xyz velocity of the target
        """
        return self.__xyz_vel

    def get_quat(self) -> np.ndarray:
        """
        :return: quaternion orientation of the target
        """
        return self.__quat

    def get_quat_vel(self) -> np.ndarray:
        """
        :return: quaternion velocity of the target
        """
        return np.asarray(self.__quat_vel)

    def get_abg(self) -> np.ndarray:
        """
        :return: euler angles of the target
        """
        return np.asarray(quat2euler(self.__quat))

    def get_abg_vel(self) -> np.ndarray:
        """
        :return: euler angles velocity of the target
        """
        return np.asarray(quat2euler(self.__quat_vel))

    def get_gripper_state(self) -> GripperState:
        """
        :return: gripper state of the target
        """
        return self.__gripper_state

    def set_xyz(self, xyz: np.ndarray) -> None:
        """
        :param xyz: xyz position of the target
        """
        assert len(xyz) == 3
        self.__xyz = np.asarray(xyz)

    def set_xyz_vel(self, xyz_vel: np.ndarray) -> None:
        """
        :param xyz_vel: xyz velocity of the target
        """
        assert len(xyz_vel) == 3
        self.__xyz_vel = np.asarray(xyz_vel)

    def set_quat(self, quat: np.ndarray) -> None:
        """
        :param quat: quaternion orientation of the target
        """
        assert len(quat) == 4
        self.__quat = np.asarray(quat)

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
        :return: x position of the target
        """
        return self.__xyz[1]

    @x.setter
    def x(self, val: float) -> None:
        """
        :param val: x position of the target
        """
        tmp = self.get_xyz()
        tmp[0] = val
        self.set_xyz(tmp)

    @property
    def y(self) -> np.ndarray:
        """
        :return: y position of the target
        """
        return self.__xyz[1]

    @y.setter
    def y(self, val: float) -> None:
        """
        :param val: y position of the target
        """
        tmp = self.get_xyz()
        tmp[1] = val
        self.set_xyz(tmp)

    @property
    def z(self) -> np.ndarray:
        """
        :return: z position of the target
        """
        return self.__xyz[2]

    @z.setter
    def z(self, val: float) -> None:
        """
        :param val: z position of the target
        """
        tmp = self.get_xyz()
        tmp[2] = val
        self.set_xyz(tmp)

    @property
    def gripper_state(self) -> GripperState:
        """
        :return: gripper state of the target
        """
        return self.__gripper_state

    @gripper_state.setter
    def gripper_state(self, val: GripperState) -> None:
        """
        :param val: gripper state of the target
        """
        self.__gripper_state = val

    def check_ob(self, x_bounds, y_bounds, z_bounds, clip=False) -> bool:
        """
        Check if the target is out of bounds and clip it if necessary.
        Only positions and not orientations or velocities are checked.

        :param x_bounds:
        :param y_bounds:
        :param z_bounds:
        :param clip: whether to clip the target or not

        :return: True if the target is out of bounds, False otherwise
        """
        ob = False
        if self.x < x_bounds[0]:
            if clip:
                self.x = x_bounds[0]
            ob = True

        if self.x > x_bounds[1]:
            if clip:
                self.x = x_bounds[1]
            ob = True

        if self.y < y_bounds[0]:
            if clip:
                self.y = y_bounds[0]
            ob = True

        if self.y > y_bounds[1]:
            if clip:
                self.y = y_bounds[1]
            ob = True

        if self.z < z_bounds[0]:
            if clip:
                self.z = z_bounds[0]
            ob = True

        if self.z > z_bounds[1]:
            if clip:
                self.z = z_bounds[1]
            ob = True

        return ob



