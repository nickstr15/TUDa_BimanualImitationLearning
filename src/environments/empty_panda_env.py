from typing import Dict, Tuple
import numpy as np

from src.environments.core.panda_environment import PandaEnvBase

class EmptyPandaEnv(PandaEnvBase):
    """
    Empty environment with two Panda robots.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(
            scene_file="dual_panda_env.xml",
            **kwargs
        )

    @property
    def _default_free_joint_positions(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Default free joint positions for the environment.
        The empty environment has no free joints and therefor returns an empty dictionary.
        :return:
        """
        return {}

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
    env = EmptyPandaEnv()
    env.visualize_static()
    env.close()


