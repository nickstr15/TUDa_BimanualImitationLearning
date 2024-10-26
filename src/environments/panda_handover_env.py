from typing import Dict, Tuple
import numpy as np
import mujoco
from src.environments.core.panda_environment import PandaEnvBase

class PandaHandoverEnv(PandaEnvBase):
    """
    Environment with two Panda robots to perform a handover of a cuboid
    and place the cuboid in a box.

    cuboid size: 5x20x5cm
    inner box size: 20x40x6cm
    """
    _target_tolerance = np.array([
        0.025, 0.025, 1e-3
    ])

    def __init__(self, **kwargs):
        super().__init__(
            scene_file="dual_panda_handover.xml",
            **kwargs
        )

    @property
    def _default_free_joints_quat_pos(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        return {
            "cuboid_position" : (
                np.array([1.0, 0.0, 0.0, 0.0]),
                np.array([0.4, -0.4, 0])
            ),
            "box_position" : (
                np.array([1.0, 0.0, 0.0, 0.0]),
                np.array([0.4, 0.4, 0])
            )
        }

    def _get_random_free_joints_quat_pos(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError # TODO



    def _get_obs(self) -> Dict:
        return {}

    def _get_object_positions(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Get the current positions of the cuboid and the box.
        :return: dict with positions and quaternion of the cuboid and the box
        """
        body_names = ["cuboid_center", "box_center"]
        positions = {}
        for name in body_names:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            xpos = self.data.xpos[body_id]
            xquat = self.data.xquat[body_id]
            positions[name] = (xpos, xquat)

        return positions

    def _get_info(self) -> Dict:
        return {}

    def _get_reward(self) -> float:
        return 0.0

    def _get_terminated(self) -> bool:
        return self._check_success()

    def _get_truncated(self) -> bool:
        return False

    def _check_success(self) -> bool:
        object_positions = self._get_object_positions()
        cuboid_position = object_positions["cuboid_center"][0]
        box_position = object_positions["box_center"][0]

        # Check if the cuboid is in the box by using target tolerance
        is_done = np.all(np.abs(cuboid_position - box_position) <= self._target_tolerance)

        return is_done


if __name__ == "__main__":
    env = PandaHandoverEnv()
    env.visualize_static()
    env.close()



