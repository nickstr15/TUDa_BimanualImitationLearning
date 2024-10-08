import time
import numpy as np

from src.control.utils.enums import GripperState
from src.control.utils.target import Target
from src.environments import PandaBimanualHandoverEnv

class PandaBimanualHandoverDemo(PandaBimanualHandoverEnv):
    def run(self) -> None:
        self.set_state(qpos=self.q_home, qvel=np.zeros_like(self.data.qvel))

        targets = self.x_home_targets

        targets["panda_01"].set_xyz(targets["panda_01"].get_xyz())
        targets["panda_02"].set_xyz(targets["panda_02"].get_xyz())

        targets["panda_01"].set_gripper_state(GripperState.CLOSED)

        while True:
            ctrl = self._generate_control(targets)
            self.do_simulation(ctrl, self.frame_skip)

            self.render()

if __name__ == "__main__":
    env = PandaBimanualHandoverDemo()
    env.run()

