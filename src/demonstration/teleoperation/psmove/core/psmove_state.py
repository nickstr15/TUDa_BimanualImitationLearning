from enum import IntEnum

import numpy as np
from transforms3d.euler import euler2quat

class PSMoveTarget(IntEnum):
    LEFT = 0
    RIGHT = 1
    UNKNOWN = 2

class PSMoveButtonState(IntEnum):
    RELEASED = 0
    NOW_PRESSED = 1
    STILL_PRESSED = 2

class PsMoveState:
    def __init__(self, serial, color=(0, 0, 0), target=PSMoveTarget.UNKNOWN):
        self._color = color
        self._serial = serial
        self._target = target

        self._btn_square = PSMoveButtonState.RELEASED
        self._btn_triangle = PSMoveButtonState.RELEASED
        self._btn_circle = PSMoveButtonState.RELEASED
        self._btn_cross = PSMoveButtonState.RELEASED
        self._btn_select = PSMoveButtonState.RELEASED
        self._btn_start = PSMoveButtonState.RELEASED
        self._btn_t = PSMoveButtonState.RELEASED
        self._btn_move = PSMoveButtonState.RELEASED
        self._btn_ps = PSMoveButtonState.RELEASED

        self._trigger = 0

        self._pos = np.array([0.0, 0.0, 0.0])
        self._quat = euler2quat(0.0, 0.0, 0.0)

    @property
    def color(self) -> tuple:
        return self._color

    @property
    def serial(self) -> str:
        return self._serial

    @property
    def target(self) -> PSMoveTarget:
        return self._target

    @property
    def btn_square(self) -> PSMoveButtonState:
        return self._btn_square

    @btn_square.setter
    def btn_square(self, value : PSMoveButtonState):
        self._btn_square = value

    @property
    def btn_triangle(self) -> PSMoveButtonState:
        return self._btn_triangle

    @btn_triangle.setter
    def btn_triangle(self, value : PSMoveButtonState):
        self._btn_triangle = value

    @property
    def btn_circle(self) -> PSMoveButtonState:
        return self._btn_circle

    @btn_circle.setter
    def btn_circle(self, value : PSMoveButtonState):
        self._btn_circle = value

    @property
    def btn_cross(self) -> PSMoveButtonState:
        return self._btn_cross

    @btn_cross.setter
    def btn_cross(self, value : PSMoveButtonState):
        self._btn_cross = value

    @property
    def btn_select(self) -> PSMoveButtonState:
        return self._btn_select

    @btn_select.setter
    def btn_select(self, value : PSMoveButtonState):
        self._btn_select = value

    @property
    def btn_start(self) -> PSMoveButtonState:
        return self._btn_start

    @btn_start.setter
    def btn_start(self, value : PSMoveButtonState):
        self._btn_start = value

    @property
    def btn_t(self) -> PSMoveButtonState:
        return self._btn_t

    @btn_t.setter
    def btn_t(self, value : PSMoveButtonState):
        self._btn_t = value

    @property
    def btn_move(self) -> PSMoveButtonState:
        return self._btn_move

    @btn_move.setter
    def btn_move(self, value : PSMoveButtonState):
        self._btn_move = value

    @property
    def btn_ps(self) -> PSMoveButtonState:
        return self._btn_ps

    @btn_ps.setter
    def btn_ps(self, value: PSMoveButtonState):
        self._btn_ps = value

    @property
    def trigger(self) -> int:
        return self._trigger

    @trigger.setter
    def trigger(self, value : int):

        self._trigger = np.clip(value, 0, 255)

    @property
    def pos(self) -> np.ndarray:
        return self._pos

    @pos.setter
    def pos(self, value : np.ndarray):
        self._pos = value

    @property
    def quat(self) -> np.ndarray:
        return self._quat

    @quat.setter
    def quat(self, value : np.ndarray):
        self._quat = value

    def update_btn_state(self, btn : str, pressed : bool) -> None:
        assert btn in ["square", "triangle", "circle", "cross", "select", "start", "t", "move", "ps"]

        btn_state = getattr(self, f"_btn_{btn}")

        if pressed:
            btn_state = PSMoveButtonState.NOW_PRESSED if btn_state == PSMoveButtonState.RELEASED else PSMoveButtonState.STILL_PRESSED
        else:
            btn_state = PSMoveButtonState.RELEASED

        setattr(self, f"_btn_{btn}", btn_state)

