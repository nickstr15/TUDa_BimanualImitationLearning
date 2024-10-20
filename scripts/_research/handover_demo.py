import os
import time
from typing import List, Tuple, Dict
import numpy as np
from copy import deepcopy as copy
import argparse

from src.environments.core.action import OSAction
from src.utils.real_time import RealTimeHandler
from src.utils.record_video import export_video

from src.control.utils.enums import GripperState
from src.control.utils.arm_state import ArmState
from src.environments import PandaHandoverEnv

from src.utils.paths import RECORDING_DIR

class PandaBimanualHandoverDemo(PandaHandoverEnv):
    """
    Panda bimanual handover demo environment.
    The robot arms follow a hardcoded trajectory to perform a handover task.
    """
    def _build_targets_traj(self) -> List[Tuple[Dict[str, ArmState], float]]:
        """
        Builds a trajectory of targets and their durations
        to complete the handover task

        :return: trajectory of targets and their duration in seconds
        """
        targets_traj = []

        # (1) Move robot to home position
        targets = self.x_home_targets
        targets_traj += [(copy(targets), 1.0)]

        # (2) Move panda_02 to cuboid
        targets["panda_02"].set_xyz(np.array([0.4, -0.43, 0.025])) # cuboid at 0.4, -0.4
        targets_traj += [(copy(targets), 4.0)]


        # (3) Close gripper of panda_02
        targets["panda_02"].set_gripper_state(GripperState.CLOSED)
        targets_traj += [(copy(targets), 1.0)]

        # (4) Move panda_02 up
        targets["panda_02"].set_xyz(np.array([0.4, -0.43, 0.3]))
        targets_traj += [(copy(targets), 3.0)]

        # (5) Move panda_02 to handover position
        # and panda_01 close to handover position
        targets["panda_01"].set_xyz(np.array([0.3, 0.05, 0.5]))
        targets["panda_01"].set_quat(np.array([1, 1, 0, 0]))
        targets["panda_02"].set_xyz(np.array([0.25, 0, 0.5]))
        targets["panda_02"].set_quat(np.array([-1, 1, 0, 0]))
        targets_traj += [(copy(targets), 5.0)]

        # (6) Move panda_01 to cuboid in handover position
        targets["panda_01"].set_xyz(np.array([0.3, 0.0, 0.5]))
        targets_traj += [(copy(targets), 1.0)]

        # (7) Close gripper of panda_01
        targets["panda_01"].set_gripper_state(GripperState.CLOSED)
        targets_traj += [(copy(targets), 1.0)]

        # (8) Open gripper of panda_02
        targets["panda_02"].set_gripper_state(GripperState.OPEN)
        targets_traj += [(copy(targets), 1.0)]

        # (9) Move panda_02 back
        targets["panda_02"].set_xyz(np.array([0.3, -0.1, 0.5]))
        targets_traj += [(copy(targets), 1.0)]

        # (10) Move to home position
        targets = self.x_home_targets
        targets["panda_01"].set_gripper_state(GripperState.CLOSED)
        targets_traj += [(copy(targets), 1.0)]

        # (11) Move panda_01 to box
        targets["panda_01"].set_xyz(np.array([0.4, 0.42, 0.08]))
        #targets["panda_01"].set_quat(np.array([0, 0.4472136, 0.8944272, 0]))
        targets_traj += [(copy(targets), 5.0)]

        # (12) Open gripper of panda_01
        targets["panda_01"].set_gripper_state(GripperState.OPEN)
        targets_traj += [(copy(targets), 5.0)]

        total_duration = sum([duration for _, duration in targets_traj])
        print(f"Trajectory with total (real-time) duration: {total_duration}sec")

        return targets_traj

    def run(self, logging : bool = False):
        """
        Run the handover demo.

        :param logging: boolean value indicating if progress is logged to the console.
        :return:
        """

        #minimum number of steps in done state
        min_steps_terminated = int(1.0 * self.render_fps)
        steps_terminated = 0

        trajectory = self._build_targets_traj()

        rt = RealTimeHandler(self.render_fps)
        rt.reset()
        for i, (targets, duration) in enumerate(trajectory):

            if logging:
                print(f"Step {i+1}/{len(trajectory)}: {duration}sec")
            steps = int(duration * self.render_fps)

            for _ in range(steps):
                #######################################################
                # Actual simulation step ##############################
                _, _, terminated, _, _ = self.step(OSAction(targets)) #
                self.render()                                         #
                #######################################################

                if terminated:
                    steps_terminated += 1
                    if steps_terminated >= min_steps_terminated:
                        print("Done.")
                        return
                else:
                    steps_terminated = 0

                if self.render_mode == "human":
                    rt.sleep()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Handover demo")
    parser.add_argument("--record", "-r", action="store_true", help="Record video of the demo")
    args = parser.parse_args()
    recording = args.record

    env = PandaBimanualHandoverDemo(
        render_mode="rgb_array" if recording else "human",
        store_frames=recording,
        # Full HD resolution (if the demo is too long, consider reducing the resolution to prevent memory overflow)
        width=1920,
        height=1080,
    )
    env.reset()
    env.run(logging=recording)

    if recording:
        print("Exporting video...")
        export_video(
            frames=env.get_mujoco_renders(),
            video_folder=os.path.join(RECORDING_DIR, "handover_demo"),
            filename="handover_demo.mp4",
            fps=env.render_fps,
        )
    env.close()

