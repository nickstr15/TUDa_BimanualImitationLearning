import sys
import os
import time

import yaml


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

    def __init__(self, frequency=20.0):
        self.dt = 1.0/frequency
        self._left_controller = None
        self._right_controller = None

    def start(self):
        self._identify_controllers()

    def _identify_controllers(self):
        move_count = psmove.count_connected()

        if move_count == 0:
            raise Exception("No controllers connected")

        for idx in range(move_count):
            controller = psmove.PSMove(idx)
            controller_serial = controller.get_serial()

            if controller_serial == LEFT_ADDRESS:
                print("Left controller found: ", controller_serial)
                self._left_controller = controller
                self._left_controller.set_leds(0, 0, 255)
                self._left_controller.update_leds()
            elif controller_serial == RIGHT_ADDRESS:
                print("Right controller found: ", controller_serial)
                self._right_controller = controller
                self._right_controller.set_leds(255, 0, 0)
                self._right_controller.update_leds()
            else:
                raise Exception(f"Unknown controller with serial: {controller_serial}." + \
                                "Please update the config file with the correct serials.")

    def stop(self):
        ...


if __name__ == "__main__":
    psmove_interface = PSMoveInterface()
    psmove_interface.start()
    time.sleep(10)
    psmove_interface.stop()







