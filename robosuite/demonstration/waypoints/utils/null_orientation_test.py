from typing import OrderedDict
import robosuite as suite
from robosuite.utils.transform_utils import mat2euler, quat2mat

from robosuite.demonstration.waypoints.core.waypoint_expert import TwoArmWaypointExpertBase

class NullOrientationTest(TwoArmWaypointExpertBase):
    """
    Specific waypoint expert for the TwoArmPickPlace environment.
    """

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
            "initial_state_right_null_orientation": self.__initial_state_right_null_orientation,
            "initial_state_left_null_orientation": self.__initial_state_left_null_orientation,
        }

        dct.update(update)
        return dct

    def __initial_state_right_null_orientation(self, obs: OrderedDict = None) -> dict:
        """
        Initial state for the right arm with null orientation.

        :param obs: observation after reset
        :return: dictionary with the target position, orientation, and gripper state
        """
        right_dct = self._initial_state_right(obs)
        print("HOME EULER RIGHT: ", mat2euler(quat2mat(right_dct["quat"])))
        right_dct["quat"] = self._null_quat_right
        return right_dct

    def __initial_state_left_null_orientation(self, obs: OrderedDict = None) -> dict:
        """
        Initial state for the left arm with null orientation.

        :param obs: observation after reset
        :return: dictionary with the target position, orientation, and gripper state
        """
        left_dct = self._initial_state_left(obs)
        print("HOME EULER LEFT: ", mat2euler(quat2mat(left_dct["quat"])))
        left_dct["quat"] = self._null_quat_left
        return left_dct

def run_null_orientation_test(robots: str | list[str]):
    """
    Run the null orientation test for the given robot config.
    :param robots:
    :return:
    """
    test_env = suite.make(
        env_name="TwoArmEmpty",
        robots=robots,
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )

    expert = NullOrientationTest(
        test_env,
        waypoints_file="two_arm_null_orientation_check.yaml",
    )
    expert.visualize(1)

if __name__ == "__main__":
    robot_cfg = ["H1ArmsOnly"]
    run_null_orientation_test(robot_cfg)

