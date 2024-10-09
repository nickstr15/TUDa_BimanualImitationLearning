import time
import numpy as np

from src.control.utils.enums import GripperState
from src.control.utils.target import Target
from src.environments import BasePandaBimanualEnv

class MoveToPointPandaBimanualEnv(BasePandaBimanualEnv):

    def visualize(self, duration=10) -> None:
        self._reset()

        targets = self.x_home_targets

        targets["panda_01"].set_xyz(targets["panda_01"].get_xyz() + np.array([0.1, 0, 0]))
        targets["panda_02"].set_xyz(targets["panda_02"].get_xyz() + np.array([-0.1, 0, 0]))

        targets["panda_01"].set_gripper_state(GripperState.CLOSED)

        start_time = time.time()
        while time.time() - start_time < duration:
            ctrl = self._generate_control(targets)
            self.do_simulation(ctrl, self.frame_skip)

            self.render()

    def visualize_relative(self, duration=20) -> None:
        # Move robot to home position
        self.reset()

        # Define initial targets
        targets = {
            "panda_01": Target(),
            "panda_02": Target(),
        }

        def set_targets(m):
            offset = 0.01 if m else -0.01
            abg_offset = np.pi / 3 if m else -np.pi / 3

            targets["panda_01"].set_xyz(np.array([0, offset, offset]))
            targets["panda_02"].set_xyz(np.array([0, -offset, -offset]))

            targets["panda_01"].set_gripper_state(GripperState.CLOSED if m else GripperState.OPEN)
            targets["panda_02"].set_gripper_state(GripperState.OPEN if m else GripperState.CLOSED)

            targets["panda_01"].set_abg(np.array([0, 0, -abg_offset]))
            targets["panda_02"].set_abg(np.array([0, 0, abg_offset]))

        set_targets(1)  # Initialize with mode 1

        start_time = time.time()
        switch_time = start_time
        switch_after = 2
        mode = 1

        while time.time() - start_time < duration:
            # Generate and apply control
            ctrl = self._generate_control(targets, relative_targets=True)
            self.do_simulation(ctrl, self.frame_skip)
            self.render()

            # Switch targets after the defined interval
            if time.time() - switch_time > switch_after:
                mode = 1 - mode  # Toggle between 0 and 1
                set_targets(mode)
                switch_time = time.time()

if __name__ == "__main__":
    env = MoveToPointPandaBimanualEnv()

    #env.visualize(3)
    env.visualize_relative(10)

    env.close()

