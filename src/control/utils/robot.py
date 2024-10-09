import copy
import time
from enum import Enum
from threading import Lock
from typing import Any, Dict, List, Callable, Tuple

import mujoco
import numpy as np

from src.control.utils.device import Device
from src.control.utils.enums import DeviceState, RobotState

class Robot:
    """
    The Robot class encapsulates the robot parameters and is the interface to mujoco
    """
    def __init__(
        self,
        sub_devices: List[Device],
        robot_name,
        model,
        data,
        use_sim,
        collect_hz=1000
    ) -> None:
        """
        :param sub_devices: list of devices connected to the robot
        :param robot_name: name of the robot
        :param model: mujoco model
        :param data: mujoco data
        :param use_sim: boolean value indicating if simulation is being used
        :param collect_hz: data collection frequency
        """
        self._data = data
        self._model = model

        self._sub_devices = sub_devices
        self._sub_devices_dict: Dict[str, Device] = dict()
        for dev in self._sub_devices:
            self._sub_devices_dict[dev.name] = dev

        self._name = robot_name
        self._num_scene_joints = self._model.nv

        self._all_joint_ids = np.array([], dtype=np.int32)
        for dev in self._sub_devices:
            self._all_joint_ids = np.hstack([self._all_joint_ids, dev.all_joint_ids])
        self._all_joint_ids = np.sort(np.unique(self._all_joint_ids))

        self._num_joints_total = len(self._all_joint_ids)

        self._data_collect_hz = collect_hz
        self.__use_sim = use_sim
        self.__running = False

        self.__state_locks: Dict[RobotState, Lock] = dict([(key, Lock()) for key in RobotState])
        self.__state_var_map: Dict[RobotState, Callable[[], np.ndarray]] = {
            RobotState.M: lambda: self.__get_M(),
            RobotState.DQ: lambda: self.__get_dq(),
            RobotState.J: lambda: self.__get_jacobian(),
            RobotState.G: lambda: self.__get_gravity(),
        }
        self.__state: Dict[RobotState, Any] = dict()

    @property
    def name(self) -> str:
        """
        Get the name of the robot
        :return: the name of the robot
        """
        return self._name

    @property
    def sub_devices(self) -> List[Device]:
        """
        Get the sub devices of the robot
        :return: the sub devices of the robot as a list
        """
        return self._sub_devices if (type(self._sub_devices) == list) else list(self._sub_devices)

    @property
    def sub_devices_dict(self) -> Dict[str, Device]:
        """
        Get the sub devices of the robot
        :return: the sub devices of the robot as a dictionary
        """
        return self._sub_devices_dict

    @property
    def all_joint_ids(self) -> np.ndarray:
        """
        Get the joint ids of the robot
        :return: the joint ids of the robot
        """
        return self._all_joint_ids

    @property
    def num_joints_total(self) -> int:
        """
        Get the total number of joints of the robot
        :return: the total number of joints of the robot
        """
        return self._num_joints_total

    def __get_gravity(self) -> np.ndarray:
        """
        :return: gravity vector
        """
        return self._data.qfrc_bias

    def __get_jacobian(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Return the Jacobians for all devices,
        so that OSC can stack them according to the provided the target entries

        :return: tuple of Jacobians and their indices
        """
        Js = dict()
        J_idxs = dict()
        start_idx = 0
        for name, device in self._sub_devices_dict.items():
            J_sub = device.get_state(DeviceState.J)
            J_idxs[name] = np.arange(start_idx, start_idx + J_sub.shape[0])
            start_idx += J_sub.shape[0]
            J_sub = J_sub[:, self._all_joint_ids]
            Js[name] = J_sub
        return Js, J_idxs

    def __get_dq(self) -> np.ndarray:
        """
        Get the joint velocities of all the devices

        :return: joint velocities for all devices
        """
        dq = np.zeros(self._all_joint_ids.shape)
        for dev in self._sub_devices:
            dq[dev.all_joint_ids] = dev.get_state(DeviceState.DQ)
        return dq

    def __get_M(self) -> np.ndarray:
        """
        Get the mass matrix of the robot

        :return: mass matrix of the robot
        """
        M = np.zeros((self._num_scene_joints, self._num_scene_joints))
        mujoco.mj_fullM(self._model, M, self._data.qM)
        M = M[np.ix_(self._all_joint_ids, self._all_joint_ids)]
        return M

    def get_state(self, state_var: RobotState) -> Any:
        """
        Get the state of the robot of the given state variable.
        :param state_var: the state variable to get

        :return: the state of the robot
        """
        if self.__use_sim:
            func = self.__state_var_map[state_var]
            state = copy.copy(func())
        else:
            self.__state_locks[state_var].acquire()
            state = copy.copy(self.__state[state_var])
            self.__state_locks[state_var].release()
        return state

    def __set_state(self, state_var: RobotState) -> None:
        """
        Set the state of the robot corresponding to the key value (if exists)

        :param state_var: the state variable to set
        :return:
        """
        assert self.__use_sim is False
        self.__state_locks[state_var].acquire()
        func = self.__state_var_map[state_var]
        value = func()
        # Make sure to copy (or else reference will stick to Dict value)
        self.__state[state_var] = copy.copy(value)
        self.__state_locks[state_var].release()

    def is_running(self) -> bool:
        """
        Check if the robot is running
        :return: boolean value indicating if the robot is running
        """
        return self.__running

    def is_using_sim(self) -> bool:
        """
        Check if the robot is using simulation
        :return: boolean value indicating if the robot is using simulation
        """
        return self.__use_sim

    def __update_state(self) -> None:
        """
        Update the state of the robot
        :return:
        """
        assert self.__use_sim is False
        for var in RobotState:
            self.__set_state(var)

    def start(self) -> None:
        """
        Start the robot
        :return:
        """
        assert self.__running is False and self.__use_sim is False
        self.__running = True
        interval = float(1.0 / float(self._data_collect_hz))
        prev_time = time.time()
        while self.__running:
            for dev in self._sub_devices:
                dev.update_state()
            self.__update_state()
            curr_time = time.time()
            diff = curr_time - prev_time
            delay = max(interval - diff, 0)
            time.sleep(delay)
            prev_time = curr_time

    def stop(self) -> None:
        """
        Stop the robot
        :return:
        """
        assert self.__running is True and self.__use_sim is False
        self.__running = False

    def get_device(self, device_name: str) -> Device:
        """
        Get the device with the given name
        :param device_name:
        :return: device with the given name
        """
        return self._sub_devices_dict[device_name]

    def get_all_states(self) -> Dict:
        """
        Get the state of all the devices connected plus the robot states

        :return: dictionary of all the states for sub devices and robot
        """
        state = {}
        for device_name, device in self._sub_devices_dict.items():
            state[device_name] = device.get_all_states()

        for key in RobotState:
            state[key] = self.get_state(key)

        return state

    def get_device_states(self) -> Dict:
        """
        Get the state of all the devices connected

        :return: dictionary of all the states for sub devices
        """
        state = {}
        for device_name, device in self._sub_devices_dict.items():
            state[device_name] = device.get_all_states()
        return state
