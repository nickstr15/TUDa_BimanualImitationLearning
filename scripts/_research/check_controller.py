import time
import numpy as np
from transforms3d.quaternions import axangle2quat, qmult

from src.control.utils.enums import GripperState
from src.environments import EmptyPandaEnv
from src.environments.core.action import OSAction


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

        quat = targets["panda_02"].get_quat()
        delta_quat = axangle2quat([0, 0, 1], np.deg2rad(45))
        targets["panda_02"].set_quat(qmult(delta_quat, quat))

        quat = targets["panda_01"].get_quat()
        delta_quat = axangle2quat([0, 0, 1], np.deg2rad(-45))
        targets["panda_01"].set_quat(qmult(delta_quat, quat))

        action = OSAction(targets)

        num_render_frames = int(duration * self.render_fps)
        for _ in range(num_render_frames):
            start_time = time.time()
            self.step(action)

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

