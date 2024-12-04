"""
Driver class for QWERTZ-Keyboard controller.
"""

import numpy as np
from pynput.keyboard import Controller, Key, Listener

from robosuite.devices import Device, Keyboard
from robosuite.utils.transform_utils import rotation_matrix
from typing_extensions import override


class QWERTZKeyboard(Keyboard):
    """
    A minimalistic driver class for a QWERTZ-Keyboard.
    """

    @staticmethod
    @override
    def _display_controls():
        """
        Method to pretty print controls.
        """

        def print_command(char, info):
            char += " " * (30 - len(char))
            print("{}\t{}".format(char, info))

        print("")
        print_command("Keys", "Command")
        print_command("Ctrl+q", "reset simulation")
        print_command("spacebar", "toggle gripper (open/close)")
        print_command("up-right-down-left", "move horizontally in x-y plane")
        print_command(".-,", "move vertically")
        print_command("o-p", "rotate (yaw)")
        print_command("z-h", "rotate (pitch)")
        print_command("e-r", "rotate (roll)")
        print_command("b", "toggle arm/base mode (if applicable)")
        print_command("s", "switch active arm (if multi-armed robot)")
        print_command("=", "switch active robot (if multi-robot environment)")
        print("")

    @override
    def on_press(self, key):
        """
        Key handler for key presses.
        Args:
            key (str): key that was pressed
        """

        try:
            # controls for moving position
            if key == Key.up:
                self.pos[0] -= self._pos_step * self.pos_sensitivity  # dec x
            elif key == Key.down:
                self.pos[0] += self._pos_step * self.pos_sensitivity  # inc x
            elif key == Key.left:
                self.pos[1] -= self._pos_step * self.pos_sensitivity  # dec y
            elif key == Key.right:
                self.pos[1] += self._pos_step * self.pos_sensitivity  # inc y
            elif key.char == ".":
                self.pos[2] -= self._pos_step * self.pos_sensitivity  # dec z
            elif key.char == ",":
                self.pos[2] += self._pos_step * self.pos_sensitivity  # inc z

            # controls for moving orientation
            elif key.char == "e":
                drot = rotation_matrix(angle=0.1 * self.rot_sensitivity, direction=[1.0, 0.0, 0.0])[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates x
                self.raw_drotation[1] -= 0.1 * self.rot_sensitivity
            elif key.char == "r":
                drot = rotation_matrix(angle=-0.1 * self.rot_sensitivity, direction=[1.0, 0.0, 0.0])[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates x
                self.raw_drotation[1] += 0.1 * self.rot_sensitivity
            elif key.char == "z":
                drot = rotation_matrix(angle=0.1 * self.rot_sensitivity, direction=[0.0, 1.0, 0.0])[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates y
                self.raw_drotation[0] += 0.1 * self.rot_sensitivity
            elif key.char == "h":
                drot = rotation_matrix(angle=-0.1 * self.rot_sensitivity, direction=[0.0, 1.0, 0.0])[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates y
                self.raw_drotation[0] -= 0.1 * self.rot_sensitivity
            elif key.char == "p":
                drot = rotation_matrix(angle=0.1 * self.rot_sensitivity, direction=[0.0, 0.0, 1.0])[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates z
                self.raw_drotation[2] += 0.1 * self.rot_sensitivity
            elif key.char == "o":
                drot = rotation_matrix(angle=-0.1 * self.rot_sensitivity, direction=[0.0, 0.0, 1.0])[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates z
                self.raw_drotation[2] -= 0.1 * self.rot_sensitivity

        except AttributeError as e:
            pass


