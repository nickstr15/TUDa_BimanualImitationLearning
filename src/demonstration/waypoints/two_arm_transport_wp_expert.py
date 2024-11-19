from typing import OrderedDict
import numpy as np

import robosuite as suite
from robosuite import TwoArmLift, TwoArmTransport
from robosuite.utils.transform_utils import quat2mat, mat2euler, axisangle2quat, quat_multiply

from src.demonstration.waypoints import TwoArmPickPlaceWaypointExpert
from src.utils.robot_targets import GripperTarget
from src.demonstration.waypoints.core.waypoint_expert import TwoArmWaypointExpertBase


class TwoArmTransportWaypointExpert(TwoArmPickPlaceWaypointExpert):
    """
    Specific waypoint expert for the TwoArmTransport environment.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._env: TwoArmTransport = self._env  # for type hinting in pycharm

    ######################################################################
    # Definition of special ee targets ###################################
    ######################################################################
    def _create_ee_target_methods_dict(self) -> dict:
        """
        Create a dictionary of methods to get the special end-effector targets.

        All methods take the observation after reset as input.

        :return: dictionary with the methods
        """
        dct = super()._create_ee_target_methods_dict()
        update = {
            # TODO
        }

        dct.update(update)
        return dct


def example(
    n_episodes: int = 10,
    robots: str | list[str] = ["Panda"]*2
):
    two_arm_transport = suite.make(
        env_name="TwoArmTransport",
        robots=robots,
        env_configuration="parallel",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )

    expert = TwoArmTransportWaypointExpert(
        environment=two_arm_transport,
        waypoints_file="two_arm_transport_wp.yaml",
    )
    expert.visualize(n_episodes)


if __name__ == "__main__":
    example()
    #example(2, ["Kinova3", "Kinova3"])
    #example(2, ["IIWA", "IIWA"])
    #example(2, ["UR5e", "UR5e"])
    #example(2, ["Panda", "IIWA"])