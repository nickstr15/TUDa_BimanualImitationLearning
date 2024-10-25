from typing import Dict, Tuple, List
import numpy as np

from src.environments.core.panda_environment import PandaEnvBase

class EmptyPandaEnv(PandaEnvBase):
    """
    Empty environment with two Panda robots as bimanual setup.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(
            scene_file="dual_panda_empty.xml",
            **kwargs
        )

    @property
    def _default_free_joints_quat_pos(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        return {} # no free objects in the scene

    def _get_random_free_joints_quat_pos(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        return {} # no free objects in the scene

    def _get_obs(self) -> Dict:
        return {}

    def _get_info(self) -> Dict:
        return {}

    def _get_reward(self) -> float:
        return 0.0

    def _get_terminated(self) -> bool:
        return False

    def _get_truncated(self) -> bool:
        return False

    def _check_success(self) -> bool:
        return False

if __name__ == "__main__":
    env = EmptyPandaEnv()
    env.visualize_static()
    env.close()


