import copy
from enum import Enum
from threading import Lock
from typing import Any, Callable, Dict

import mujoco
import numpy as np


class DeviceState(Enum):
    Q = "Q"
    Q_ACTUATED = "Q_ACTUATED"
    DQ = "DQ"
    DQ_ACTUATED = "DQ_ACTUATED"
    DDQ = "DDQ"
    EE_XYZ = "EE_XYZ"
    EE_XYZ_VEL = "EE_XYZ_VEL"
    EE_QUAT = "EE_QUAT"
    FORCE = "FORCE"
    TORQUE = "TORQUE"
    J = "JACOBIAN"


class Device:
    """
    A Device is a single Panda arm.
    The Device class encapsulates the device parameters,
    and it collects data from the simulator, obtaining the
    desired device states.
    """
    def __init__(self, device_config: Dict, model, data, use_sim: bool):
        self._data = data
        self._model = model
        self.__use_sim = use_sim

        self._name = device_config["name"]
        self._max_vel = device_config.get("max_vel")
        self._EE = device_config["EE"]

        self._num_gripper_joints = device_config["num_gripper_joints"]

        if "start_body" in device_config.keys():
            start_body_name = device_config["start_body"]
            start_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, start_body_name)
        else:
            start_body = 0

        # Reference: ABR Control
        # Get the joint ids, using the specified EE / start body
        # start with the end-effector (EE) and work back to the world body
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self._EE)
        joint_ids = []
        joint_names = []

        while model.body_parentid[body_id] != 0 and model.body_parentid[body_id] != start_body:
            jntadrs_start = model.body_jntadr[body_id]
            tmp_ids = []
            tmp_names = []
            for ii in range(model.body_jntnum[body_id]):
                tmp_ids.append(jntadrs_start + ii)
                joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, tmp_ids[-1])
                tmp_names.append(joint_name)
            joint_ids += tmp_ids[::-1]
            joint_names += tmp_names[::-1]
            body_id = model.body_parentid[body_id]

        # Flip the list so it starts with the base of the arm / first joint
        self._joint_names = joint_names[::-1]
        self._joint_ids = np.array(joint_ids[::-1])

        gripper_start_idx = self._joint_ids[-1] + 1
        self._gripper_ids = np.arange(
            gripper_start_idx, gripper_start_idx + self._num_gripper_joints
        )
        self._all_joint_ids = np.hstack([self._joint_ids, self._gripper_ids])

        # Find the actuator and control indices
        actuator_trnids = model.actuator_trnid[:, 0]
        self._ctrl_idxs = np.intersect1d(actuator_trnids, self._all_joint_ids, return_indices=True)[1]
        self._actuator_trnids = actuator_trnids[self._ctrl_idxs]

        # Initialize dicts to keep track of the state variables and locks
        self.__state_var_map: Dict[DeviceState, Callable[[], np.ndarray]] = {
            DeviceState.Q: lambda: data.qpos[self._all_joint_ids],
            DeviceState.Q_ACTUATED: lambda: data.qpos[self._joint_ids],
            DeviceState.DQ: lambda: data.qvel[self._all_joint_ids],
            DeviceState.DQ_ACTUATED: lambda: data.qvel[self._joint_ids],
            DeviceState.DDQ: lambda: data.qacc[self._all_joint_ids],
            DeviceState.EE_XYZ: lambda: data.xpos[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self._EE)
            ],
            DeviceState.EE_XYZ_VEL: lambda: data.cvel[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self._EE), :3
            ],
            DeviceState.EE_QUAT: lambda: data.xquat[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self._EE)
            ],
            DeviceState.FORCE: lambda: self.__get_force(),
            DeviceState.TORQUE: lambda: self.__get_torque(),
            DeviceState.J: lambda: self.__get_jacobian(),
        }

        self.__state: Dict[DeviceState, Any] = dict()
        self.__state_locks: Dict[DeviceState, Lock] = dict([(key, Lock()) for key in DeviceState])

        # These are the keys we should use when returning data from get_all_states()
        self._concise_state_vars = [
            DeviceState.Q_ACTUATED,
            DeviceState.DQ_ACTUATED,
            DeviceState.EE_XYZ,
            DeviceState.EE_XYZ_VEL,
            DeviceState.EE_QUAT,
            DeviceState.FORCE,
            DeviceState.TORQUE,
        ]

    @property
    def name(self):
        return self._name

    @property
    def all_joint_ids(self):
        return self._all_joint_ids

    @property
    def max_vel(self):
        return self._max_vel

    @property
    def ctrl_idxs(self):
        return self._ctrl_idxs

    @property
    def actuator_trnids(self):
        return self._actuator_trnids

    def __get_jacobian(self):
        """
        :returns: The full jacobian (of the Device, using its EE)
        """
        _J = np.array([])
        _EE_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, self._EE)
        _J = np.zeros((3, self._model.nv))
        _Jr = np.zeros((3, self._model.nv))
        mujoco.mj_jacBody(self._model, self._data, jacp=_J, jacr=_Jr, body=_EE_id)
        _J = np.vstack([_J, _Jr]) if _J.size else _Jr
        return _J

    def __get_R(self):
        """
        Get rotation matrix for device's ft_frame
        """
        if self._name == "ur5right": #TODO replace
            site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, "ft_frame_ur5right")
        elif self._name == "ur5left":
            site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, "ft_frame_ur5left")
        else:
            raise ValueError(f"Wrong name specification: {self._name}")

        xmat = self._data.site_xmat[site_id].reshape(3, 3)
        return xmat

    def __get_force(self):
        """
        Get the external forces, used (for admittance control) acting upon
        the gripper sensors
        """
        if self._name == "ur5right": # TODO replace and fix indices
            force = np.matmul(self.__get_R(), self._data.sensordata[0:3])
            return force
        if self._name == "ur5left":
            force = np.matmul(self.__get_R(), self._data.sensordata[6:9])
            return force
        else:
            return np.zeros(3)

    def __get_torque(self):
        """
        Get the external torques, used (for admittance control) acting upon
        the gripper sensors
        """
        if self._name == "ur5right": #TODO replace + fix incides
            force = np.matmul(self.__get_R(), self._data.sensordata[3:6])
            return force
        if self._name == "ur5left":
            force = np.matmul(self.__get_R(), self._data.sensordata[9:12])
            return force
        else:
            return np.zeros(3)

    def __set_state(self, state_var: DeviceState):
        """
        Set the state of the device corresponding to the key value (if exists)
        """
        assert self.__use_sim is False
        self.__state_locks[state_var].acquire()

        ########################################
        # NOTE: This is only a placeholder for non-simulated/mujoco devices
        # To get a realtime value, you'd need to communicate with a device driver/api
        # Then set the var_value to what the device driver/api returns
        var_func = self.__state_var_map[state_var]
        var_value = var_func()
        ########################################

        # (for Mujoco) Make sure to copy (or else reference will stick to Dict value)
        self.__state[state_var] = copy.copy(var_value)
        self.__state_locks[state_var].release()

    def get_state(self, state_var: DeviceState):
        """
        Get the state of the device corresponding to the key value (if exists)
        """
        if self.__use_sim:
            func = self.__state_var_map[state_var]
            state = copy.copy(func())
        else:
            self.__state_locks[state_var].acquire()
            state = copy.copy(self.__state[state_var])
            self.__state_locks[state_var].release()
        return state

    def get_all_states(self):
        return dict([(key, self.get_state(key)) for key in self._concise_state_vars])

    def update_state(self):
        """
        This should be running in a thread, e.g. Robot.start()
        """
        assert self.__use_sim is False
        for var in DeviceState:
            self.__set_state(var)