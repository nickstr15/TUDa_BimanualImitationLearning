import numpy as np
from transforms3d.euler import quat2euler
from transforms3d.quaternions import quat2mat, axangle2quat, qmult

from src.demonstration.waypoints.core.waypoint_expert import TwoArmWaypointExpertBase
from src.environments.manipulation.two_arm_pick_place import TwoArmPickPlace

import robosuite as suite


class TwoArmPickPlaceWaypointExpert(TwoArmWaypointExpertBase):
    """
    Panda handover expert agent that follows a predefined trajectory of waypoints.
    """

    def __init__(
            self,
            env : TwoArmPickPlace,
            **kwargs):
        """
        Constructor for the PandaHandoverWpExpertBase class.
        :param dual_panda_env_args: additional arguments for the environment, default is None
        :param kwargs: additional arguments for the expert, see WaypointExpert (core.waypoint_expert)
        """
        super().__init__(
            environment=env,
            waypoints_file="two_arm_pick_place_wp.yaml",
            **kwargs
        )

    ######################################################################
    # Definition of special ee targets ###################################
    ######################################################################

    # rotation of the end-effector in the world frame that aligns with cuboid rotation = 0°
    _zero_quat_ee = np.array([0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0])

    def _create_ee_target_methods_dict(self) -> dict:
        """
        Create a dictionary of methods to get the special end-effector targets.
        """
        return {
            "initial_state_left": self.__initial_state_left,
            "initial_state_right": self.__initial_state_right

            "pre_pick_up_robot_left": self.__pre_pick_up_robot_left,
            "pre_drop_off_robot_right": self.__pre_drop_off_robot_right,
        }

    def __initial_state_left(self) -> dict:
        """
        Initial state for the left arm.
        """
        raise NotImplementedError

    def __initial_state_right(self) -> dict:
        """
        Initial state for the right arm.
        """
        raise NotImplementedError

    def __pre_pick_up_robot_left(self) -> dict:
        """
        Pre-pick up position for the second arm.
        """
        raise NotImplementedError

    def __pre_drop_off_robot_right(self) -> dict:
        """
        Pre-drop off position for the right arm.
        """
        raise NotImplementedError

    def __get_target(self, name: str, offset: np.ndarray, grip : float) -> dict:
        """
        Get the target position and orientation for the end-effector.
        :param name: name of the body
        :param offset: offset from the body
        :param grip: gripper state

        :return: dictionary with the target position, orientation, and gripper state
        """
        return {
            "pos": np.array([0.0, 0.0, 0.0]),
            "quat": np.array([1.0, 0.0, 0.0, 0.0]),
            "grip": grip
        }

        # TODO: Implement the following code snippet
        """
        quat, pos = ..., ...

        # the objects are only rotated around the z-axis
        _, _, angle = np.rad2deg(quat2euler(quat))
        # normalize the angle to the range -180° to 180°
        angle = (angle + 180) % 360 - 180
        # map it to its symmetrical equivalent in the range -90° to 90°
        angle = angle - 180 if angle > 90 else (
            angle + 180 if angle < -90 else angle)
        # back to radians
        angle = np.deg2rad(angle)
        quat = axangle2quat([0, 0, 1], angle)

        ee_quat = qmult(quat, self._zero_quat_ee)
        ee_pos = pos + np.dot(quat2mat(quat), offset)

        return {
            "pos": ee_pos,
            "quat": ee_quat,
            "grip": grip
        }
        """

if __name__ == "__main__":
    two_arm_pick_place = suite.make(
        env_name="TwoArmPickPlace",
        robots=["Panda", "Panda"],
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )

    expert = TwoArmPickPlaceWaypointExpert(two_arm_pick_place)
    expert.visualize(10)
    expert.dispose()