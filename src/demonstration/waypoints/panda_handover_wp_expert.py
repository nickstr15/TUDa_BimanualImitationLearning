import os

from src.demonstration.waypoints.core.waypoint import Waypoint
from src.demonstration.waypoints.core.waypoint_expert import WaypointExpertBase
from src.environments import PandaHandoverEnv
from src.environments.core.enums import ActionMode


class PandaHandoverWpExpert(WaypointExpertBase):
    """
    Panda handover expert agent that follows a predefined trajectory of waypoints.
    """
    def __init__(
            self,
            dual_panda_env_args : dict = None,
            **kwargs):
        """
        Constructor for the PandaHandoverWpExpertBase class.
        :param dual_panda_env_args: additional arguments for the environment, default is None
        :param kwargs: additional arguments for the expert, see WaypointExpert (core.waypoint_expert)
        """
        super().__init__(
            environment=PandaHandoverEnv(**dual_panda_env_args if dual_panda_env_args else {}),
            waypoints_file="panda_handover_wp.yaml",
            **kwargs
        )

if __name__ == "__main__":
    env_args = dict(
        visualize_targets=True,
        action_mode = ActionMode.RELATIVE
    )

    expert = PandaHandoverWpExpert(env_args)
    expert.visualize()
    expert.dispose()