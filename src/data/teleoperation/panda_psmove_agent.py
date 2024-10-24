from typing import Type

import numpy as np

from src.data.teleoperation.core.psmove_agent import PSMoveAgentBase
from src.environments.core.enums import ActionMode
from src.environments.core.panda_environment import PandaEnvBase
from src.environments.core.viewer_config import CamConfig


class PandaPSMoveAgent(PSMoveAgentBase):
    """
    Class for teleoperation in a PandaEnvironment with PSMove controllers.
    """
    def __init__(
        self,
        env_class : Type[PandaEnvBase],
        panda_env_args : dict = None,
        **kwargs
    ) -> None:
        """
        Constructor for the PandaPSMoveAgent class.
        :param env_class: the class of the panda environment, e.g. PandaHandoverEnv
        :param panda_env_args: arguments for the panda environment, default is None
        :param kwargs: additional arguments for the PSMoveAgentBase, see PSMoveAgentBase (core.psmove_agent)
        """
        if "left_controller_target" in kwargs:
            print("[Warning] left_controller_target is set by default to panda_01")
            kwargs.pop("left_controller_target")
        if "right_controller_target" in kwargs:
            print("[Warning] right_controller_target is set by default to panda_02")
            kwargs.pop("right_controller_target")

        env = env_class(**panda_env_args if panda_env_args else {})
        super().__init__(
            environment=env,
            left_controller_target="panda_01",
            right_controller_target="panda_02",
             **kwargs
        )

if __name__ == "__main__":
    from src.environments import EmptyPandaEnv
    env_args = dict(
        visualize_targets=True,
        action_mode = ActionMode.RELATIVE,
        width = 1280,
        height = 720
    )

    agent = PandaPSMoveAgent(EmptyPandaEnv, env_args)
    agent.run()
    agent.dispose()

