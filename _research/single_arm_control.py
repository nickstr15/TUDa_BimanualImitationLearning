import numpy as np
import robosuite as suite
from robosuite.utils.transform_utils import quat2axisangle, axisangle2quat, quat_multiply, quat_inverse

from src.utils.robot_states import EEState
from src.utils.robot_targets import EETarget, GripperTarget


def _generate_control(target, current):
    """
    Generate a control action to reach the target.
    :param target: target to reach
    :param current: current state
    :return: control action
    """
    # Position control
    pos_control = target.xyz - current.xyz

    # Orientation control
    # Get the quaternion difference
    quat_diff = quat_multiply(target.quat, quat_inverse(current.quat))
    ori_control = quat2axisangle(quat_diff)

    grip_control = GripperTarget.CLOSED_VALUE

    control = np.concatenate([pos_control, ori_control, [grip_control]])

    return control


def delta_control():
    env = suite.make(
        "PickPlace",
        robots="Panda",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
    )

    obs = env.reset()
    current = EEState(
        xyz=obs["robot0_eef_pos"],
        quat=obs["robot0_eef_quat"],
        grip=obs["robot0_gripper_qpos"],
    )

    delta_aa = np.array([0.0, np.deg2rad(30.0), 0])
    delta_quat = axisangle2quat(delta_aa)
    target_quat = quat_multiply(delta_quat, current.quat)

    target = EETarget(
        xyz=current.xyz + np.array([0.0, 0.0, 0.1]),
        quat=target_quat,
        grip=GripperTarget.OPEN_VALUE,
    )

    for _ in range(300):
        action = _generate_control(target, current)
        obs, _, _, _ = env.step(action)
        current = EEState(
            xyz=obs["robot0_eef_pos"],
            quat=obs["robot0_eef_quat"],
            grip=obs["robot0_gripper_qpos"],
        )

if __name__ == "__main__":
    delta_control()

