from typing import Dict, Tuple
import numpy as np
from transforms3d.quaternions import axangle2quat, quat2mat
import mujoco
from src.environments.core.panda_environment import PandaEnvBase

class PandaHandoverEnv(PandaEnvBase):
    """
    Environment with two Panda robots to perform a handover of a cuboid
    and place the cuboid in a box.

    cuboid size: 5x20x5cm
    inner box size: 20x40x6cm
    """
    _cuboid_size = np.array([0.05, 0.2, 0.05])
    _box_size = np.array([0.2, 0.4, 0.06])

    def __init__(self, **kwargs):
        super().__init__(
            scene_file="dual_panda_handover.xml",
            **kwargs
        )

    @property
    def _default_free_joints_quat_pos(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        return {
            "cuboid_position" : (
                np.array([1.0, 0.0, 0.0, 0.0]),
                np.array([0.4, -0.4, 0])
            ),
            "box_position" : (
                np.array([1.0, 0.0, 0.0, 0.0]),
                np.array([0.4, 0.4, 0])
            )
        }

    def _get_random_free_joints_quat_pos(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        # random orientation of the cuboid and the box
        angle_range = np.deg2rad(170)
        rot_vec = np.array([0, 0, 1])
        angle_cuboid = np.random.uniform(-angle_range, angle_range)
        angle_box = np.random.uniform(-angle_range, angle_range)
        quat_cuboid = axangle2quat(rot_vec, angle_cuboid, True)
        quat_box = axangle2quat(rot_vec, angle_box, True)

        # random position of the cuboid and the box
        x_range = [0.3, 0.5]
        y_range_cuboid = [-0.5, -0.3]
        y_range_box = [0.3, 0.5]
        x_cuboid = np.random.uniform(*x_range)
        y_cuboid = np.random.uniform(*y_range_cuboid)
        x_box = np.random.uniform(*x_range)
        y_box = np.random.uniform(*y_range_box)

        pos_cuboid = np.array([x_cuboid, y_cuboid, 0])
        pos_box = np.array([x_box, y_box, 0])

        return {
            "cuboid_position" : (quat_cuboid, pos_cuboid),
            "box_position" : (quat_box, pos_box)
        }


    def _get_obs(self) -> Dict:
        return {}

    def _get_object_quat_pos(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Get the current positions of the cuboid and the box.
        :return: dict with positions and quaternion of the cuboid and the box
        """
        body_names = ["cuboid_center", "box_center"]
        positions = {}
        for name in body_names:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            pos = self.data.xpos[body_id]
            quat = self.data.xquat[body_id]
            positions[name] = (quat, pos)

        return positions

    def _get_info(self) -> Dict:
        return {}

    def _get_reward(self) -> float:
        return 0.0

    def _get_terminated(self) -> bool:
        return self._check_success()

    def _get_truncated(self) -> bool:
        return False

    def _check_success(self) -> bool:
        object_positions = self._get_object_quat_pos()
        cuboid_quat, cuboid_pos = object_positions["cuboid_center"]
        box_quat, box_pos = object_positions["box_center"][0]

        # check if the cuboid is in correct height
        if np.abs(cuboid_pos[2] - box_pos[2]) > 0.005:
            return False

        # check if the cuboid is in the box, respect the orientation of both objects
        # Compute the corner coordinates of the cuboid and the box in the xy-plane
        def get_corners(center, quat, size):
            # Rotation matrix from quaternion
            rotation_matrix = quat2mat(quat)[:2, :2]  # Only x and y
            half_size = size[:2] / 2

            # Define local corners (relative to the center)
            corners = np.array([
                [-half_size[0], -half_size[1]],
                [-half_size[0], half_size[1]],
                [half_size[0], -half_size[1]],
                [half_size[0], half_size[1]]
            ])

            # Rotate and translate corners to global position
            rotated_corners = (rotation_matrix @ corners.T).T + center[:2]
            return rotated_corners

        # Get cuboid and box corner positions
        cuboid_corners = get_corners(cuboid_pos, cuboid_quat, self._cuboid_size)
        box_corners = get_corners(box_pos, box_quat, self._box_size)

        # Find box boundaries in the x-y plane
        box_x_min, box_x_max = box_corners[:, 0].min(), box_corners[:, 0].max()
        box_y_min, box_y_max = box_corners[:, 1].min(), box_corners[:, 1].max()

        # Check if all cuboid corners are within the box boundaries
        cuboid_in_box = all(
            (box_x_min <= corner[0] <= box_x_max) and (box_y_min <= corner[1] <= box_y_max)
            for corner in cuboid_corners
        )

        # Determine if the task is completed
        return cuboid_in_box


if __name__ == "__main__":
    env = PandaHandoverEnv()
    env.visualize_static()
    env.close()



