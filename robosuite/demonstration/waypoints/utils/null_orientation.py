import os
import yaml
import numpy as np

import robosuite as suite
from robosuite.models.robots.robot_model import REGISTERED_ROBOTS
from robosuite.utils.transform_utils import euler2mat, mat2quat


class NullOrientationContainer:
    """
    Container for the null orientation for the robot arms.

    It returns the null orientations as quaternions [x, y, z, w].
    """
    def __init__(self):
        yaml_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "null_orientation.yaml"
        )

        with open(yaml_file, "r") as file:
            self._list = yaml.safe_load(file)

    def __getitem__(self, item):
        entry = None
        for e in self._list:
            if e["name"] == item:
                entry = e
                break

        if entry is None:
            raise ValueError(f"Null orientation for {item} not found. Please specify it in null_orientation.yaml")

        match entry.get("type", "single"):
            case "single":
                return self.__to_quat(entry["null_euler"])
            case "dual":
                return self.__to_quat(entry["null_euler_right"]), self.__to_quat(entry["null_euler_left"])
            case _:
                raise ValueError(f"Unknown type {entry['type']}")

    @staticmethod
    def __to_quat(euler):
        return mat2quat(euler2mat(np.array(euler)))

def get_two_arm_null_orientation(robots: list[str] | str) -> tuple[np.array(float), np.array(float)]:
    """
    Get the null orientation for the left and right arm for a specific robot configuration.

    :return: null orientation for the left and right arm, as quaternions [x, y, z, w]
    """
    if isinstance(robots, str):
        robots = [robots]

    assert isinstance(robots, list) and len(robots) in [1, 2], "Expected a list of 2 robots or a single robot"

    num_arms = sum([
        len(REGISTERED_ROBOTS[rbt].arms) for rbt in robots
    ])
    assert num_arms == 2, f"Expected 2 arms, got {num_arms}"

    container = NullOrientationContainer()
    if len(robots) == 1:
        return container[robots[0]]

    return container[robots[0]], container[robots[1]]
