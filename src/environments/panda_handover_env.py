from typing import Dict, Tuple
import numpy as np

from src.environments.core.panda_environment import PandaEnvBase

class PandaHandoverEnv(PandaEnvBase):
    """
    Environment with two Panda robots to perform a handover of a cuboid.
    """

    def __init__(self, **kwargs):
        super().__init__(
            scene_file="dual_panda_handover_env.xml",
            **kwargs
        )

    @property
    def _free_joints(self):
        return ["free_joint_cuboid"]

    @property
    def _default_free_joint_positions(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Default free joint positions for the bimanual handover environment.

        :return: orientation and position of the free joint cuboid
        """
        return {
            "free_joint_cuboid" : (
                np.array([1.0, 0.0, 0.0, 0.0]),
                np.array([0.4, -0.4, 0.15])
            )
        }

    def _get_obs(self) -> Dict:
        return {
            "qpos": self.data.qpos,
            "qvel": self.data.qvel,
        }

    def _get_info(self) -> Dict:
        return {}

    def _get_reward(self) -> float:
        return 0.0

    def _get_terminated(self) -> bool:
        return False

    def _get_truncated(self) -> bool:
        return False


if __name__ == "__main__":
    env = PandaHandoverEnv()
    env.visualize_static(3)
    env.close()



