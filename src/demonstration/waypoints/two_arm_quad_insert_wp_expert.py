import numpy as np
import robosuite as suite
from robosuite.utils.transform_utils import quat2mat, quat2axisangle

from src.demonstration.waypoints.two_arm_lift_wp_expert import TwoArmLiftWaypointExpert
from src.environments.manipulation.two_arm_quad_insert import TwoArmQuadInsert
from src.utils.robot_targets import GripperTarget


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
            "pre_goal_robot_right": self.__pre_goal_robot_right,
            "pre_goal_robot_left": self.__pre_goal_robot_left,

            "goal_robot_right": self.__goal_robot_right,
            "goal_robot_left": self.__goal_robot_left,
        }

        dct.update(update)
        return dct

    def __pre_goal_robot_right(self, obs: dict = None) -> dict:
        """
        Pre-goal position for the right arm.

        :param obs: observation after reset
        :return: dictionary with the target position, orientation, and gripper state
        """
        handle0_xpos = obs["handle0_xpos"]
        handle1_xpos = obs["handle1_xpos"]
        v_to_handle = self._env.bracket.center_to_handle0 if handle0_xpos[1] < handle1_xpos[1] \
            else self._env.bracket.center_to_handle1

        offset_xpos = np.array([0, 0, 0.05])
        return self.__calculate_goal_pose(obs, v_to_handle, offset_xpos)

    def __pre_goal_robot_left(self, obs: dict = None) -> dict:
        """
        Pre-goal position for the left arm.

        :param obs: observation after reset
        :return: dictionary with the target position, orientation, and gripper state
        """
        handle0_xpos = obs["handle0_xpos"]
        handle1_xpos = obs["handle1_xpos"]
        v_to_handle = self._env.bracket.center_to_handle0 if handle0_xpos[1] >= handle1_xpos[1] \
            else self._env.bracket.center_to_handle1

        offset_xpos = np.array([0, 0, 0.05])
        return self.__calculate_goal_pose(obs, v_to_handle, offset_xpos)

    def __goal_robot_left(self, obs: dict = None) -> dict:
        """
        Feedback goal position for the left arm.

        :param obs: observation after reset
        :return: dictionary with the target position, orientation, and gripper state
        """
        handle0_xpos = obs["handle0_xpos"]
        handle1_xpos = obs["handle1_xpos"]
        v_to_handle = self._env.bracket.center_to_handle0 if handle0_xpos[1] >= handle1_xpos[1] \
            else self._env.bracket.center_to_handle1

        return self.__calculate_goal_pose(obs, v_to_handle)

    def __goal_robot_right(self, obs: dict = None) -> dict:
        """
        Feedback goal position for the right arm.

        :param obs: observation after reset
        :return: dictionary with the target position, orientation, and gripper state
        """
        handle0_xpos = obs["handle0_xpos"]
        handle1_xpos = obs["handle1_xpos"]
        v_to_handle = self._env.bracket.center_to_handle0 if handle0_xpos[1] < handle1_xpos[1] \
            else self._env.bracket.center_to_handle1

        return self.__calculate_goal_pose(obs, v_to_handle)

    def __calculate_goal_pose(
            self,
            obs: dict,
            v_to_handle: np.ndarray,
            offset_xpos: np.ndarray = np.zeros(3),
    ) -> dict:
        """
        Calculate the pre-goal pose for the robot.

        :param obs: observation after reset
        :param v_to_handle: vector from the center of the bracket to the handle
        :param offset_xpos: additional offset in the world frame
        :return: dictionary with the target position, orientation, and gripper state
        """
        target_xpos = np.array(obs["target_xpos"])
        target_quat = np.array(obs["target_quat"])

        # convert to world frame
        v_to_handle = quat2mat(target_quat) @ v_to_handle

        target_xpos += v_to_handle

        return self._calculate_pose(
            xpos=target_xpos,
            quat=target_quat,
            null_quat=self._null_quat_right,
            offset_xpos=offset_xpos,
            grip=GripperTarget.CLOSED_VALUE,
        )


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
    example()
    #example(2, ["Kinova3", "Kinova3"])
    #example(2, ["IIWA", "IIWA"])
    #example(2, ["UR5e", "UR5e"])
    #example(2, ["Panda", "IIWA"])