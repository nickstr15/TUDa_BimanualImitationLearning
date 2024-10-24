import sys
import os
import threading
import time

import numpy as np
import yaml
from transforms3d.euler import quat2euler, euler2quat

from src.data.teleoperation.core.psmove_state import PsMoveState, PSMoveTarget, PSMoveButtonState

######################################################################################
# Setup PSMove #######################################################################
######################################################################################
full_config_path = os.path.join(os.path.dirname(__file__), "psmove_config.yml")
with open(full_config_path, 'r') as f:
    PSMOVE_CONFIG = yaml.safe_load(f)

PSMOVE_API_PATH = PSMOVE_CONFIG["psmove_api_path"]
LEFT_ADDRESS = PSMOVE_CONFIG["left_controller_address"]
RIGHT_ADDRESS = PSMOVE_CONFIG["right_controller_address"]

sys.path.insert(0, PSMOVE_API_PATH)

import psmove
######################################################################################

class PSMoveInterface:
    """
    Interface for collecting raw data from PSMove controllers.

    The interface will automatically identify connected controllers and start collecting data from them.
    The controller states are stored in PsMoveState objects.

    The interface will also calibrate the controllers and set the LED colors, corresponding to the psmove_config.yaml.
    Left controller will have red LEDs, right controller will have blue LEDs,
    and any other controller will have white LEDs.

    Example usage:
    psmove_interface = PSMoveInterface()
    psmove_interface.start()

    # Do something with the controller states

    psmove_interface.stop()
    """

    _buttons = ["square", "triangle", "circle", "cross", "select", "start", "t", "move", "ps"]
    _left_address = LEFT_ADDRESS
    _right_address = RIGHT_ADDRESS

    def __init__(self, frequency : float = 20.0):
        """
        Initialize the PSMoveInterface.
        :param frequency: frequency of the data collection in Hz
        """
        self._dt = 1.0 / frequency
        self._controllers = {}
        self._controller_workers = {}
        self._controller_states = {}
        self._tracker = None
        self._running = False

    def start(self) -> None:
        self._identify_controllers()
        self._initialize_tracker()

        self._run_calibration()

        self._start_collecting()

    def _identify_controllers(self) -> None:
        move_count = psmove.count_connected()
        if move_count == 0:
            raise Exception("No controllers connected via Bluetooth")

        for idx in range(move_count):
            controller = psmove.PSMove(idx)
            while not controller.poll(): pass
            controller_serial = controller.get_serial()
            battery = controller.get_battery() / 5.0 * 100.0

            if not controller.connection_type == psmove.Conn_Bluetooth:
                print("[WARNING]\tSkipping controller with non bluetooth connection: ", controller_serial)
                continue

            if controller_serial == LEFT_ADDRESS:
                print("[INFO]\tLeft controller found: ", controller_serial, " Battery: ", battery, "%")
                rgb = (255, 0, 0)
                target = PSMoveTarget.LEFT
            elif controller_serial == RIGHT_ADDRESS:
                print("[INFO]\tRight controller found: ", controller_serial, " Battery: ", battery, "%")
                rgb = (0, 0, 255)
                target = PSMoveTarget.RIGHT
            else:
                print("[INFO]\tUnknown controller found: ", controller_serial, " Battery: ", battery, "%")
                rgb = (255, 255, 255)
                target = PSMoveTarget.UNKNOWN

            controller.enable_orientation(True)
            self._controllers[controller_serial] = controller

            state = PsMoveState(controller_serial, rgb, target)
            self._controller_states[controller_serial] = state

            self._controller_workers[controller_serial] = threading.Thread(
                target=self._collect_state,
                args=(controller, state)
            )

            controller.set_leds(*rgb)
            controller.update_leds()

    def _initialize_tracker(self) -> None:
        self._tracker = psmove.PSMoveTracker()
        self._tracker.set_mirror(True)

    def _run_calibration(self) -> None:
        for controller, state in zip(self._controllers.values(), self._controller_states.values()):
            while not controller.poll(): pass
            serial = controller.get_serial()
            print("[INFO]\tCalibrating controller: ", serial, end="\r")

            result = -1
            start_time = time.time()
            while result != psmove.Tracker_CALIBRATED:
                elapsed = np.round(time.time() - start_time, 2)
                print("[INFO]\tCalibrating controller: ", serial, " (", elapsed, "s)", end="\r")
                result = self._tracker.enable_with_color(controller, *state.color)

            print("[INFO]\tcontroller.has_calibration(): ", controller.has_calibration())

            controller.reset_orientation()
            controller.set_leds(0, 255, 0) # green when calibrated
            controller.update_leds()
            print("[INFO]\tCalibration complete for controller: ", serial)
            time.sleep(1)

    def _start_collecting(self) -> None:
        self._running = True
        for worker in self._controller_workers.values():
            worker.start()

    def _collect_state(self, controller : psmove.PSMove, state : PsMoveState) -> None:

        controller.reset_orientation()

        while self._running:
            while not controller.poll(): pass
            self._tracker.update_image()
            self._tracker.update()

            buttons = controller.get_buttons()
            state.update_btn_state("square", buttons & psmove.Btn_SQUARE)
            state.update_btn_state("triangle", buttons & psmove.Btn_TRIANGLE)
            state.update_btn_state("circle", buttons & psmove.Btn_CIRCLE)
            state.update_btn_state("cross", buttons & psmove.Btn_CROSS)
            state.update_btn_state("select", buttons & psmove.Btn_SELECT)
            state.update_btn_state("start", buttons & psmove.Btn_START)
            state.update_btn_state("t", buttons & psmove.Btn_T)
            state.update_btn_state("move", buttons & psmove.Btn_MOVE)
            state.update_btn_state("ps", buttons & psmove.Btn_PS)

            if state.btn_t == PSMoveButtonState.NOW_PRESSED:
                controller.reset_orientation()

            if self._tracker.get_status(controller) == psmove.Tracker_TRACKING:
                x, y, radius = self._tracker.get_position(controller)
                state.pos = np.array([x, y, radius])

            quat = controller.get_orientation()
            eul0, eul1, eul2 = quat2euler(quat)
            new_eul = [eul2, -eul0, eul1]
            new_quat = euler2quat(*new_eul)
            state.quat = np.asarray(new_quat)

            trigger_value = controller.get_trigger()
            state.trigger = trigger_value



            self._on_update(state)

            time.sleep(self._dt)

    def _on_update(self, state : PsMoveState) -> None:
        if state.btn_ps == PSMoveButtonState.NOW_PRESSED or state.btn_ps == PSMoveButtonState.STILL_PRESSED:
            print("[INFO]\tInterrupting PSMoveInterface")
            self.stop()

    def stop(self):
        self._running = False
        print("[INFO]\tStopping PSMoveInterface")

if __name__ == "__main__":
    psmove_interface = PSMoveInterface()
    psmove_interface.start()

    while psmove_interface._running:
        time.sleep(0.1)







