from typing import Dict, Tuple, List

import numpy as np
from transforms3d.derivations.quaternions import qmult
from transforms3d.euler import quat2euler
from transforms3d.quaternions import qinverse

from src.control.utils.enums import RobotState, DeviceState
from src.control.utils.device import Device
from src.control.utils.robot import Robot
from src.control.utils.target import Target
from src.control.control_configs.controller_config import ControllerConfig

class OSCGripperController:
    """
    OSC provides Operational Space Control for a given Robot.
    This controller accepts targets as an input, and generates a control signal
    for the devices that are linked to the targets.

    The control signals are the following:
        - torque signals for the joints controlling the EE
        - position signals in [0, 255] for the gripper
    """
    def __init__(
        self,
        robot: Robot,
        input_device_configs: List[Tuple[str, Dict]],
        nullspace_config: Dict = None,
        use_g=True,
        admittance_gain=0,
    ):
        """
        Initialize the OSC Controller with separate gripper control

        :param robot: the robot object
        :param input_device_configs: the device configurations
        :param nullspace_config: the nullspace controller configuration
        :param use_g: boolean to use gravity compensation
        :param admittance_gain: gain for the admittance controller, if <= 0, admittance is not used
        """
        self.robot = robot

        # Create a dict, device_configs, which maps a device name to a
        # ControllerConfig. ControllerConfig is a lightweight wrapper
        # around the dict class to add some desired methods
        self.device_names = [device_cfg[0] for device_cfg in input_device_configs]
        self.device_configs = dict()
        for device_cfg in input_device_configs:
            self.device_configs[device_cfg[0]] = ControllerConfig(device_cfg[1])
        self.nullspace_config = nullspace_config
        self.use_g = use_g
        self.admittance = admittance_gain > 0
        self.admittance_gain = admittance_gain

        self.osc_mask = []
        for device in self.robot.sub_devices:
            self.osc_mask += [True] * len(device.joint_ids) + [False] * len(device.gripper_ids)

        # Obtain the controller configuration parameters
        # and calculate the task space gains
        for device_name in self.device_configs.keys():
            kv, kp, ko = self.device_configs[device_name].get_params(["kv", "kp", "ko"])
            task_space_gains = np.array([kp] * 3 + [ko] * 3)
            self.device_configs[device_name]["task_space_gains"] = task_space_gains
            self.device_configs[device_name]["lamb"] = task_space_gains / kv

    def __Mx(self, J, M) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the inverse of the task space inertia matrix

        :param J: Jacobian matrix
        :param M: inertia matrix

        :return: Mx, M_inv
        """
        M_inv = self.__svd_solve(M)
        Mx_inv = np.dot(J, np.dot(M_inv, J.T))
        threshold = 1e-4
        if abs(np.linalg.det(Mx_inv)) >= threshold:
            Mx = self.__svd_solve(Mx_inv)
        else:
            Mx = np.linalg.pinv(Mx_inv, rcond=threshold * 0.1)
        return Mx, M_inv

    @staticmethod
    def __svd_solve(A):
        """
        Use the SVD Method to calculate the inverse of a matrix
        Parameters
        ----------
        A: Matrix
        """
        u, s, v = np.linalg.svd(A)
        A_inv = np.dot(v.transpose(), np.dot(np.diag(s**-1), u.transpose()))
        return A_inv

    def __limit_vel(self, u_task: np.ndarray, device: Device) -> np.ndarray:
        """
        Limit the velocity of the task space control vector

        :param u_task: array of length 6 corresponding to the task space control

        :return: the limited task space control vector
        """
        if device.max_vel is not None:
            kv, kp, ko, lamb = self.device_configs[device.name].get_params(
                ["kv", "kp", "ko", "lamb"]
            )
            scale = np.ones(6)

            # Apply the sat gains to the x,y,z components
            norm_xyz = np.linalg.norm(u_task[:3])
            sat_gain_xyz = device.max_vel[0] / kp * kv
            scale_xyz = device.max_vel[0] / kp * kv
            if norm_xyz > sat_gain_xyz:
                scale[:3] *= scale_xyz / norm_xyz

            # Apply the sat gains to the a,b,g components
            norm_abg = np.linalg.norm(u_task[3:])
            sat_gain_abg = device.max_vel[1] / ko * kv
            scale_abg = device.max_vel[1] / ko * kv
            if norm_abg > sat_gain_abg:
                scale[3:] *= scale_abg / norm_abg

            u_task = kv * scale * lamb * u_task
        else:
            print("Device max_vel must be set in the config file (yaml)!")
            raise Exception

        return u_task

    @staticmethod
    def calc_task_space_error(target: Target, device: Device) -> np.ndarray:
        """
        Compute the difference between the target and device EE
        for the x,y,z and a,b,g components

        :param target: the target object
        :param device: the device object to calculate the error for

        :return: the task space error
        """
        u_task = np.zeros(6)

        # Calculate x,y,z error
        diff = device.get_state(DeviceState.EE_XYZ) - target.get_xyz()
        u_task[:3] = diff

        # Calculate a,b,g error
        q_r = np.array(
            qmult(device.get_state(DeviceState.EE_QUAT), qinverse(target.get_quat()))
        )
        u_task[3:] = quat2euler(q_r)

        return u_task

    def generate_absolute(self, targets: Dict[str, Target]) -> Tuple[list, list]:
        """
        Generate control signal for the corresponding devices which are in the
        robot's sub-devices. Accepts a dictionary of device names (keys),
        which map to a Target (absolute target).

        :param targets: dict of device names mapping to absolute Target objects

        :return: tuple of (control indices, corresponding control signals)
        """
        return self._generate(targets)

    def generate_relative(self, relative_targets: Dict[str, Target]) -> Tuple[list, list]:
        """
        Generate control signal for the corresponding devices which are in the
        robot's sub-devices. Accepts a dictionary of device names (keys),
        which map to a Target (relative target).

        :param relative_targets: dict of device names mapping to relative Target objects

        :return: tuple of (control indices, corresponding control signals)
        """
        return self._generate(relative_targets, relative=True)


    def _generate(self, targets: Dict[str, Target], relative : bool = False) -> Tuple[list, list]:
        """
        Generate control signal for the corresponding devices which are in the
        robot's sub-devices. Accepts a dictionary of device names (keys),
        which map to a Target (absolute target).

        :param targets: dict of device names mapping to absolute Target objects
        :param relative: boolean to indicate if the targets are relative

        :return: tuple of (control indices, corresponding control signals)
        """
        if relative:
            raise NotImplementedError

        # check that all device names are in the targets dict
        for device_name in self.device_names:
            if device_name not in targets.keys():
                raise ValueError(f"Error: Must Provide a Target Value for {device_name}!")

        if self.robot.is_using_sim() is False:
            assert self.robot.is_running(), "Robot must be running!"

        robot_state = self.robot.get_all_states()

        # Get the Jacobian for the devices passed in
        Js, J_idxs = robot_state[RobotState.J]
        J = np.array([])
        for device_name in targets.keys():
            J = np.vstack([J, Js[device_name]]) if J.size else Js[device_name]

        mask = self.osc_mask
        J = J[:, mask]
        M = robot_state[RobotState.M]

        M = M[mask]
        M = M[:, mask]

        # Compute the inverse matrices used for task space operations
        Mx, M_inv = self.__Mx(J, M)

        # Initialize the control vectors and sim data needed for control calculations
        dq = robot_state[RobotState.DQ] # joint velocities
        dq = dq[mask]
        dx = np.dot(J, dq) # end effector velocities

        uv_all = np.dot(M, dq) # current joint space forces of current movement

        u_all = np.zeros(len(self.robot.all_joint_ids[mask]))
        u_task_all = np.array([])
        ext_f = np.array([])

        for device_name, target in targets.items():
            device = self.robot.get_device(device_name)

            # Calculate the error from the device EE to target
            u_task = self.calc_task_space_error(target, device)
            stiffness = np.array(self.device_configs[device_name]["k"] + [1] * 3)
            damping = np.array(self.device_configs[device_name]["d"] + [1] * 3)

            # Apply gains to the error terms
            if device.max_vel is not None:
                u_task = self.__limit_vel(u_task, device)
                u_task *= stiffness
            else:
                task_space_gains = self.device_configs[device.name]["task_space_gains"]
                u_task *= task_space_gains * stiffness

            # Apply kv gain
            kv = self.device_configs[device.name]["kv"]
            target_vel = np.hstack([target.get_xyz_vel(), target.get_abg_vel()])
            if np.all(target_vel == 0):
                ist, c1, c2 = np.intersect1d(
                    device.all_joint_ids, self.robot.all_joint_ids[mask], return_indices=True
                )
                u_all[c2] = -1 * kv * uv_all[c2]
            else:
                diff = dx[J_idxs[device_name]] - np.array(target_vel)
                u_task += kv * diff * damping

            force = np.append(
                robot_state[device_name][DeviceState.FORCE],
                robot_state[device_name][DeviceState.TORQUE],
            )
            ext_f = np.append(ext_f, force)
            u_task_all = np.append(u_task_all, u_task)

        # Transform task space signal to joint space
        if self.admittance is True:
            u_all -= np.dot(J.T, np.dot(Mx, u_task_all + self.admittance_gain * ext_f))
        else:
            u_all -= np.dot(J.T, np.dot(Mx, u_task_all))

        # Apply gravity forces
        if self.use_g:
            qfrc_bias = robot_state[RobotState.G]
            u_all += qfrc_bias[self.robot.all_joint_ids[mask]]

        # Apply the nullspace controller using the specified parameters
        # (if passed to constructor / initialized)
        if self.nullspace_config is not None:
            damp_kv = self.nullspace_config["kv"]
            u_null = np.dot(M, -damp_kv * dq)
            Jbar = np.dot(M_inv, np.dot(J.T, Mx))
            null_filter = np.eye(len(self.robot.all_joint_ids[mask])) - np.dot(J.T, Jbar.T)
            u_all += np.dot(null_filter, u_null)

        # Return the forces and indices to apply the forces
        # + gripper control
        ctrls = []
        ctrl_idxs = []
        for dev in self.robot.sub_devices:
            ist, c1, c2 = np.intersect1d(
                dev.actuator_trnids, self.robot.all_joint_ids[mask], return_indices=True
            )
            ctrls.append(u_all[c2])
            ist2, c12, c22 = np.intersect1d(dev.actuator_trnids, ist, return_indices=True)
            ctrl_idxs.append(dev.ctrl_idxs[c22])

        for i, dev in enumerate(self.robot.sub_devices):
            # add gripper control
            if dev.has_gripper:
                ctrls[i] = np.append(ctrls[i], targets[dev.name].get_gripper_state())
                ctrl_idxs[i] = np.append(ctrl_idxs[i], dev.gripper_ctrl_idx)

        return ctrl_idxs, ctrls



