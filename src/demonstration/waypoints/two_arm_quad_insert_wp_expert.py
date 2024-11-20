import robosuite as suite

from src.demonstration.waypoints.two_arm_lift_wp_expert import TwoArmLiftWaypointExpert
from src.environments.manipulation.two_arm_quad_insert import TwoArmQuadInsert


class TwoArmQuadInsertWaypointExpert(TwoArmLiftWaypointExpert):
    """
    Specific waypoint expert for the TwoArmLift environment.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._env: TwoArmQuadInsert = self._env  # for type hinting in pycharm

        self._object_name = "bracket"

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
    two_arm_pick_place = suite.make(
        env_name="TwoArmQuadInsert",
        robots=robots,
        env_configuration="parallel",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )

    expert = TwoArmQuadInsertWaypointExpert(
        environment=two_arm_pick_place,
        waypoints_file="two_arm_quad_insert_wp.yaml",
    )
    expert.visualize(n_episodes)


if __name__ == "__main__":
    #example()
    #example(2, ["Kinova3", "Kinova3"])
    example(2, ["IIWA", "IIWA"])
    #example(2, ["UR5e", "UR5e"])
    #example(2, ["Panda", "IIWA"])