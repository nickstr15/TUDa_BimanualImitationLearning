from abc import ABC
import numpy as np
from numpy.typing import NDArray
from typing import Dict

from src.control.utils.target import Target
from src.control.utils.enums import GripperState
from src.environments.core.environment_interface import IEnvironment


class PandaEnvBase(IEnvironment, ABC):
    """
    Base class for bimanual Panda environments.
    """

    def __init__(
        self,
        scene_file: str = None,
        **kwargs
    ) -> None:
        IEnvironment.__init__(
            self,
            scene_file=scene_file,
            control_config_file="dual_panda.yaml",
            robot_name="DualPanda",
            **kwargs
        )

    @property
    def q_home(self) -> NDArray[np.float64]:
        return np.array(
            [0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04] * 2,
            dtype=np.float64
        )

    @property
    def x_home_targets(self) -> Dict[str, Target]:
        targets = {
            "panda_01": Target(),
            "panda_02": Target(),
        }

        targets["panda_01"].set_xyz(np.array([0.55449948, 0.4, 0.68450243]))
        targets["panda_01"].set_quat(np.array([0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0]))
        targets["panda_01"].set_gripper_state(GripperState.OPEN)

        targets["panda_02"].set_xyz(np.array([0.55449948, -0.4, 0.68450243]))
        targets["panda_02"].set_quat(np.array([0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0]))
        targets["panda_02"].set_gripper_state(GripperState.OPEN)
        return targets