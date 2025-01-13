"""
This file implements a wrapper for visualizing the end-effector targets in a robot environment with OSC controllers.
"""

import xml.etree.ElementTree as ET

from robosuite_ext.controllers.parts.arm import OperationalSpaceController

from robosuite_ext.utils.mjcf_utils import new_body, new_geom
from robosuite_ext.utils.transform_utils import make_pose, pose_in_A_to_pose_in_B
from robosuite_ext.wrappers import Wrapper
from robosuite_ext.environments.robot_env import RobotEnv

DEFAULT_COLORS = [
    [1, 0, 0, 0.3],  # red
    [0, 0, 1, 0.3],  # blue
    [0, 1, 0, 0.3],  # green
    [1, 1, 0, 0.3],  # yellow
    [1, 0, 1, 0.3],  # magenta
    [0, 1, 1, 0.3],  # cyan
]

class TargetVisualizationWrapper(Wrapper):
    def __init__(self, env: RobotEnv):
        """
        Initialize the target visualization wrapper. Note that this automatically conducts a (hard) reset initially to make
        sure indicators are properly added to the sim model.

        Args:
            env (RobotEnv): The environment to wrap, must use OSC controllers for all end-effectors parts
        """
        super().__init__(env)

        # get all end-effector controllers
        n_arms = sum(len(robot.arms) for robot in self.env.robots)
        self._eef_controllers = self._get_eef_controllers()
        assert n_arms == len(self._eef_controllers), \
            f"[TargetVisualizationWrapper] Have {n_arms} arms, but {len(self._eef_controllers)} end-effector controllers"
        self.env.set_xml_processor(processor=self._add_indicators_to_model)

        # Conduct a (hard) reset to make sure visualization changes propagate
        reset_mode = self.env.hard_reset
        self.env.hard_reset = True
        self.reset()
        self.env.hard_reset = reset_mode

    def step(self, action):
        """
        Extends vanilla step() function call to accommodate visualization

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (OrderedDict) observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """
        ret = super().step(action)

        self._update_target_indicators()

        return ret

    def _update_target_indicators(self):
        """
        Updates the target indicators in the mujoco simulation model
        """
        for name, part_controller in self._get_eef_controllers().items():
            ee_pos = part_controller.goal_pos
            ee_ori = part_controller.goal_ori
            # convert to world frame
            origin_pose = make_pose(part_controller.origin_pos, part_controller.origin_ori)
            ee_pose = make_pose(ee_pos, ee_ori)
            world_pose = pose_in_A_to_pose_in_B(ee_pose, origin_pose)
            # get to vector
            target_pos = world_pose[:3, 3]
            self.sim.data.set_mocap_pos(name + "_indicator_body", target_pos)


    def _get_eef_controllers(self):
        _eef_controllers = {}
        for i, robot in enumerate(self.env.robots):
            for name, part_controller in robot.composite_controller.part_controllers.items():
                if name not in ["left", "right"]:
                    continue
                assert type(part_controller) == OperationalSpaceController, \
                    f"[TargetVisualizationWrapper] Only OperationalSpaceController is supported, got {type(part_controller)}"
                _eef_controllers[f"robot{i}_{name}_eef"] = part_controller
        return _eef_controllers


    def _add_indicators_to_model(self, xml):
        """
        Adds indicators to the mujoco simulation model

        Args:
            xml (string): MJCF model in xml format, for the current simulation to be loaded
        """
        root = ET.fromstring(xml)
        worldbody = root.find("worldbody")

        for i, (name, part_controller) in enumerate(self._eef_controllers.items()):
            indicator_name = name + "_indicator_body"
            if indicator_name in xml:
                continue
            rgba = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
            indicator_body = self.__default_indicator_body(indicator_name, rgba)
            worldbody.append(indicator_body)

        xml = ET.tostring(root, encoding="utf8").decode("utf8")
        return xml

    @staticmethod
    def __default_indicator_body(
        name: str,
        rgba: list[float],
    ) -> ET.Element:
        indicator_body = new_body(name=name, pos=(0, 0, -1), mocap=True)

        sphere = new_geom(
            name=name + "_indicator_sphere_geom",
            type="sphere",
            size=[0.03],
            rgba=rgba,
            contype=0,
            conaffinity=0,
            group=2,
        )

        indicator_body.append(sphere)

        return indicator_body
