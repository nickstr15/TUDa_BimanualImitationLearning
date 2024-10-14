import time
import numpy as np

from src.control.utils.enums import GripperState
from src.control.utils.target import Target
from src.environments import EmptyPandaEnv

class PandaMoveToPointEnv(EmptyPandaEnv):
    """
    Environment with two Panda robots moving to a specific (relative) point in space.
    """

    def visualize(self, duration=10) -> None:
        """
        Visualize the two Panda robots moving to a specific point in space.
        :param duration: simulation time in seconds
        :return:
        """
        # Move robot to home position
        self.reset()

        targets = self.x_home_targets

        targets["panda_01"].set_xyz(targets["panda_01"].get_xyz() + np.array([ 0.1,  0.0, -0.3]))
        targets["panda_02"].set_xyz(targets["panda_02"].get_xyz() + np.array([-0.1, -0.1, -0.1]))

        targets["panda_01"].set_gripper_state(GripperState.CLOSED)

        num_render_frames = int(duration * self.render_fps)
        for _ in range(num_render_frames):
            start_time = time.time()
            ctrl = self._generate_control(targets)
            self.do_simulation(ctrl, self.frame_skip)

            self.render()

            # Sleep to ensure real-time rendering
            end_time = time.time()
            elapsed_time = end_time - start_time
            sleep_time = 1 / self.render_fps - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)



if __name__ == "__main__":
    env = PandaMoveToPointEnv()
    env.visualize(3)
    env.close()

