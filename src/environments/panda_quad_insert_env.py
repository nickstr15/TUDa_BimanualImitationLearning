from typing import Dict, Tuple
import numpy as np
from src.environments.core.panda_environment import PandaEnvBase

class PandaQuadInsertEnv(PandaEnvBase):
    """
    Environment with two Panda robots to perform the quad insert task
    of https://bimanual-imitation.github.io/
    """

    def __init__(self, **kwargs):
        super().__init__(
            scene_file="dual_panda_quad_insert.xml",
            **kwargs
        )

    @property
    def _default_free_joint_positions(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        return {
            "grommet_position" : (#pos="0.5 0.0 0.0" quat="1.0 0 0 0.2"
                np.array([1.0, 0.0, 0.0, -0.2]),
                np.array([0.35, -0.3, 0.0])
            ),
            "peg_position" : ( #pos="0.0 0.6 0.0" quat="1.0 0 0 1.0"
                np.array([1.0, 0.0, 0.0, 0]),
                np.array([0.35, 0.2, 0])
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
        return self._check_success()

    def _get_truncated(self) -> bool:
        return False

    def _check_success(self) -> bool:
        return False


if __name__ == "__main__":
    env = PandaQuadInsertEnv()
    env.visualize_static(10)
    env.close()



