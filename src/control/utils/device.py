import copy
from threading import Lock
from typing import Any, Callable, Dict

import mujoco
import numpy as np

from src.control.utils.arm_state import ArmState
from src.control.utils.enums import GripperState, DeviceState


class Device:
    """
    The Device class encapsulates the device parameters specified in the config file
    that is passed to MujocoApp. It collects data from the simulator, obtaining the
    desired device states.
    """

    def __init__(self, device_cfg: Dict, model, data, use_sim: bool) -> None:
        self._data = data
        self._model = model
        self.__use_sim = use_sim
        # Assign config parameters
        self._name = device_cfg["name"]
        self._max_vel = device_cfg.get("max_vel")
        self._EE = device_cfg["EE"]
        self._num_gripper_joints = device_cfg["num_gripper_joints"]
        self._gripper_range_q = device_cfg["gripper_range_q"]
        self._force_sensor_idx = device_cfg.get("force_sensor_idx", -1)
        self._torque_sensor_idx = device_cfg.get("torque_sensor_idx", -1)
        self._ft_sensor_site = device_cfg.get("ft_sensor_site", None)

        self._controller_type = device_cfg["controller"]
        self._has_gripper = self._num_gripper_joints > 0


        if "start_body" in device_cfg.keys():
            start_body_name = device_cfg["start_body"]
            start_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, start_body_name)
        else:
            start_body = 0


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
        
        self._gripper_ctrl_idx = self._ctrl_idxs[-1] + 1 if self._has_gripper else None

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
            DeviceState.GRIPPER: lambda: self.__get_gripper_state()
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
            DeviceState.GRIPPER
        ]

    @property
    def name(self):
        return self._name

    @property
    def all_joint_ids(self):
        return self._all_joint_ids

    @property
    def joint_ids(self):
        return self._joint_ids

    @property
    def gripper_ids(self):
        return self._gripper_ids

    @property
    def max_vel(self):
        return self._max_vel

    @property
    def ctrl_idxs(self):
        return self._ctrl_idxs

    @property
    def actuator_trnids(self):
        return self._actuator_trnids
    
    @property
    def gripper_ctrl_idx(self):
        return self._gripper_ctrl_idx
    
    @property
    def has_gripper(self):
        return self._has_gripper

    @property
    def controller_type(self):
        return self._controller_type

    def __get_jacobian(self):
        """
        :return: full jacobian (of the Device, using its EE)
        """
        EE_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, self._EE)
        J = np.zeros((3, self._model.nv))
        Jr = np.zeros((3, self._model.nv))
        mujoco.mj_jacBody(self._model, self._data, jacp=J, jacr=Jr, body=EE_id)
        J = np.vstack([J, Jr]) if J.size else Jr
        return J

    def __get_gripper_state(self):
        """
        Get the state of the gripper joints
        """
        q_gripper = self._data.qpos[self._gripper_ids]
        eps = 0.001 #1mm tolerance for open gripper
        return GripperState.OPEN if np.all(q_gripper >= self._gripper_range_q[1] - eps) else GripperState.CLOSED

    def __get_force(self):
        """
        Get the external forces, used (for admittance control) acting upon
        the gripper sensors
        """
        if self._force_sensor_idx > -1:
            force = np.matmul(self.__get_R_ft_frame(), self._data.sensordata[self._force_sensor_idx:self._force_sensor_idx+3])
            return force
        else:
            return np.zeros(3)

    def __get_torque(self):
        """
        Get the external torques, used (for admittance control) acting upon
        the gripper sensors
        """
        if self._torque_sensor_idx > -1:
            torque = np.matmul(self.__get_R_ft_frame(), self._data.sensordata[self._torque_sensor_idx:self._torque_sensor_idx+3])
            return torque
        else:
            return np.zeros(3)

    def __get_R_ft_frame(self):
        """
        Get rotation matrix for device's ft_frame
        """
        if self._ft_sensor_site is not None:
            site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, self._ft_sensor_site)
        else:
            raise ValueError("Invalid called method. ft_sensor_site is None.")

        xmat = self._data.site_xmat[site_id].reshape(3, 3)
        return xmat

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

        :param state_var: the state variable to get

        :return: the state of the device
        """
        if self.__use_sim:
            func = self.__state_var_map[state_var]
            state = copy.copy(func())
        else:
            self.__state_locks[state_var].acquire()
            state = copy.copy(self.__state[state_var])
            self.__state_locks[state_var].release()
        return state

    def get_all_states(self) -> Dict:
        """
        Get all the states of the device
        :return: dictionary containing all the states
        """
        return dict([(key, self.get_state(key)) for key in self._concise_state_vars])

    def __str__(self):
        all_states = self.get_all_states()
        return str(all_states)

    def update_state(self):
        """
        This should run in a thread: Robot.start()
        """
        assert self.__use_sim is False
        for var in DeviceState:
            self.__set_state(var)

    def get_arm_state(self) -> ArmState:
        """
        Get the state of the arm
        :return:
        """
        pos = self.get_state(DeviceState.EE_XYZ)
        quat = self.get_state(DeviceState.EE_QUAT)
        grip = self.get_state(DeviceState.GRIPPER)

        arm_state = ArmState()
        arm_state.set_xyz(pos)
        arm_state.set_quat(quat)
        arm_state.set_gripper_state(grip)

        return arm_state



