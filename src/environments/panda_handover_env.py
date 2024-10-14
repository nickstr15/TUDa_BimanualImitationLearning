from typing import Dict, Tuple

import numpy as np
from typing_extensions import override

from src.environments import EmptyPandaEnv

class PandaHandoverEnv(EmptyPandaEnv):
    """
    Environment with two Panda robots performing a handover of a cuboid.
    """
    def __init__(
            self,
            scene_file="dual_panda_handover_env.xml",
            **kwargs
        ):

        super().__init__(scene_file, **kwargs)

    @override
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
            ),
        }

if __name__ == "__main__":
    env = PandaHandoverEnv()
    env.visualize_static(3)
    env.close()



