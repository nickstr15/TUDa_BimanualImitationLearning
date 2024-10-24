import threading
from typing import Tuple

import numpy as np
from sympy.physics.units import current
from transforms3d.quaternions import qmult, qinverse
from typing_extensions import override

from src.control.utils.arm_state import ArmState
from src.control.utils.enums import GripperState
from src.data.teleoperation.core.psmove_interface import PSMoveInterface
from src.data.teleoperation.core.psmove_state import PsMoveState, PSMoveTarget, PSMoveButtonState
from src.environments.core.action import OSAction
from src.environments.core.enums import ActionMode
from src.environments.core.environment_interface import IEnvironment
from src.utils.clipping import clip_translation, clip_quat
from src.utils.constants import MAX_DELTA_TRANSLATION, MAX_DELTA_ROTATION
from src.utils.real_time import RealTimeHandler
from src.utils.scaling import scale_translation, scale_quat


class PSMoveAgentBase(PSMoveInterface):
    """
    Agent for teleoperation in dual arm environments with PSMove controllers.
    """
    def __init__(
        self,
        environment: IEnvironment,
        left_controller_target: str,
        right_controller_target: str,
        pos_sensitivity: float = 0.001,
        quat_sensitivity: float = 1.0,
        max_delta_translation: float = MAX_DELTA_TRANSLATION,
        max_delta_rotation: float = MAX_DELTA_ROTATION,
        psmove_frequency: float = 20.0
    ) -> None:
        """
        Initialize the PSMoveAgent.
        :param environment: Environment in which the agent acts
        :param left_controller_target: Target name in mujoco for the left controller
        :param right_controller_target: Target name in mujoco for the right controller
        :param pos_sensitivity: Sensitivity for the position control
        :param quat_sensitivity: Sensitivity for the orientation control
        :param max_delta_translation: Maximum translation distance between current position and action output
        :param max_delta_rotation: Maximum rotation angle between current orientation and action output
        :param psmove_frequency: frequency of the controller queries in Hz
        """
        assert environment.render_mode == "human", \
            "Rendering mode must be human"

        super().__init__(psmove_frequency)

        self._env = environment
        self._action_mode = self._env.action_mode
        self._left_controller_target = left_controller_target
        self._right_controller_target = right_controller_target
        self._pos_sensitivity = pos_sensitivity
        self._quat_sensitivity = quat_sensitivity

        self._max_delta_translation = max_delta_translation
        self._max_delta_rotation = max_delta_rotation

        self._references_ctrls = {
            self._left_controller_target: (None, None), # (pos, quat)
            self._right_controller_target: (None, None) # (pos, quat)
        }
        self._references_env = {
            self._left_controller_target: (None, None), # (pos, quat)
            self._right_controller_target: (None, None), # (pos, quat)
        }
        self._targets_ctrls = {
            self._left_controller_target: (None, None), # (pos, quat)
            self._right_controller_target: (None, None), # (pos, quat)
        }
        self._gripper_ctrls = {
            self._left_controller_target: GripperState.OPEN,
            self._right_controller_target: GripperState.OPEN
        }

        self._interrupt = False

    @override
    def _on_update(self, state : PsMoveState) -> None:
        """
        React to the controller updates.
        """

        if state.btn_ps == PSMoveButtonState.NOW_PRESSED or state.btn_ps == PSMoveButtonState.STILL_PRESSED:
            self._interrupt = True
            return

        if state.target == PSMoveTarget.LEFT:
            _controller_target = self._left_controller_target
        elif state.target == PSMoveTarget.RIGHT:
            _controller_target = self._right_controller_target
        else:
            return

        pos_control = state.pos
        quat_control = state.quat

        if state.btn_t == PSMoveButtonState.NOW_PRESSED: #update references if T is pressed now
            pos_env = self._env.get_device_states()[_controller_target].get_xyz()
            quat_env = self._env.get_device_states()[_controller_target].get_quat()

            self._references_ctrls[_controller_target] = (pos_control, quat_control)
            self._references_env[_controller_target] = (pos_env, quat_env)

            self._targets_ctrls[_controller_target] = (pos_control, quat_control)
        elif state.btn_t == PSMoveButtonState.STILL_PRESSED: # update targets if T is pressed
            self._targets_ctrls[_controller_target] = (pos_control, quat_control)
        else: # if T is not pressed do not move
            self._targets_ctrls[_controller_target] = (None, None)

        if state.btn_cross == PSMoveButtonState.NOW_PRESSED or state.btn_cross == PSMoveButtonState.STILL_PRESSED:
            self._gripper_ctrls[_controller_target] = GripperState.CLOSED
        elif state.btn_circle == PSMoveButtonState.NOW_PRESSED or state.btn_circle == PSMoveButtonState.STILL_PRESSED:
            self._gripper_ctrls[_controller_target] = GripperState.OPEN

    def run(self) -> None:
        """
        Run the agent in the environment.
        """

        # minimum number of steps in done state
        min_steps_terminated = int(1.0 * self._env.render_fps)
        steps_terminated = 0

        # set the initial configuration
        self._env.reset()
        self._env.render()

        super().start()

        rt = RealTimeHandler(self._env.render_fps)

        current_state = self._env.get_device_states()
        rt.reset()
        while True:
            if self._interrupt:
                break

            action = self._get_action(current_state)
            _, _, terminated, _, _ = self._env.step(action)
            self._env.render()
            current_state = self._env.get_device_states()

            if terminated:
                steps_terminated += 1
                if steps_terminated >= min_steps_terminated:
                    print("Terminated.")
                    break
            else:
                steps_terminated = 0

            rt.sleep()

        super().stop()

    def dispose(self) -> None:
        """
        Dispose the expert agent.
        """
        super().stop()
        self._env.close()

    def _get_action(self, current_state: dict) -> OSAction:
        """
        Get the action to reach the waypoint.
        The output is a clipped version of the Waypoint state to
        respect self._max_delta_translation and self._max_delta_rotation.
        :param current_state: Current state of the device
        :return: Action to reach the waypoint + boolean indicating if nothing to do
        """
        action_dict = {
            device: ArmState() # init actions with zero values
            for device in current_state.keys()
        }

        for device in current_state.keys():
            state_env = current_state[device]
            current_pos = state_env.get_xyz()
            current_quat = state_env.get_quat()

            ref_pos_ctrl, ref_quat_ctrl = self._references_ctrls[device]
            ref_pos_env, ref_quat_env = self._references_env[device]
            target_pos_ctrl, target_quat_ctrl = self._targets_ctrls[device]
            grip_ctrl = self._gripper_ctrls[device]

            #set gripper state
            action_dict[device].set_gripper_state(grip_ctrl)

            # compute the absolute translation target in the environment
            if target_pos_ctrl is not None:
                raw_delta_pos = target_pos_ctrl - ref_pos_ctrl
                ref_delta_pos = scale_translation(raw_delta_pos, self._pos_sensitivity)
                target_pos_env = ref_pos_env + ref_delta_pos
            else:
                target_pos_env = current_pos

            # compute the absolute rotation target in the environment
            if target_quat_ctrl is not None:
                raw_delta_quat = qmult(target_quat_ctrl, qinverse(ref_quat_ctrl))
                ref_delta_quat = scale_quat(raw_delta_quat, self._quat_sensitivity)
                target_quat_env = qmult(ref_delta_quat, ref_quat_env)
            else:
                target_quat_env = current_quat


            # Clip the translation
            pos_delta = clip_translation(target_pos_env - current_pos, self._max_delta_translation)
            if self._action_mode == ActionMode.ABSOLUTE:
                action_dict[device].set_xyz(pos_delta + current_pos)
            elif self._action_mode == ActionMode.RELATIVE:
                action_dict[device].set_xyz(pos_delta)

            # Clip the rotation
            quat_delta = clip_quat(
                qmult(target_quat_env, qinverse(current_quat)),
                self._max_delta_rotation
            )
            if self._action_mode == ActionMode.ABSOLUTE:
                action_dict[device].set_quat(qmult(quat_delta, current_quat))
            else:
                action_dict[device].set_quat(quat_delta)

        return OSAction(action_dict)








