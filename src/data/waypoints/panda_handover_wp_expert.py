import os

from src.data.waypoints.core.waypoint_expert import WaypointExpert
from src.environments import PandaHandoverEnv


class PandaHandoverWpExpert(WaypointExpert):
    """
    Panda handover expert agent that follows a predefined trajectory of waypoints.
    """
    def __init__(
            self,
            identifier : int = 1,
            env_args : dict = None,
            **kwargs):
        """
        Constructor for the PandaHandoverWpExpert class.
        :param identifier: id of the expert agent
        :param env_args: additional arguments for the environment, default is None
        :param kwargs: additional arguments for the expert, see WaypointExpert (core.waypoint_expert)
        """
        super().__init__(
            environment=PandaHandoverEnv(**env_args if env_args else {}),
            waypoints_file=os.path.join("panda_handover", f"{str(identifier).zfill(3)}.yaml"),
            **kwargs
        )

if __name__ == "__main__":
    expert = PandaHandoverWpExpert()
    expert.visualize()
    expert.dispose()