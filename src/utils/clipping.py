import numpy as np
from transforms3d.quaternions import quat2axangle, axangle2quat


def clip_translation(vector, max_length):
    """
    Clip the translation vector to have a maximum norm.
    :param vector: Translation vector
    :param max_length: Maximum length of the translation vector
    :return: Clipped translation vector
    """
    assert max_length > 0, "max_length must be positive"

    norm = np.linalg.norm(vector)
    if norm > max_length:
        return vector / norm * max_length
    return vector

def clip_quat(quat, max_rot):
    """
    Clip the rotation quaternion to have a maximum angle.
    :param quat: Rotation quaternion [w, x, y, z]
    :param max_rot: Maximum angle of the rotation quaternion in radians
    :return: Clipped rotation quaternion
    """
    assert max_rot > 0, "max_rot must be positive"

    axis, angle = quat2axangle(quat)
    if angle > max_rot:
        quat = axangle2quat(axis, max_rot)

    return quat