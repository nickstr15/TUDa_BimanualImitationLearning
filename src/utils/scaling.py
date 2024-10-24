import numpy as np
from transforms3d.quaternions import quat2axangle, axangle2quat

def scale_translation(vector, scaling):
    """
    Scale the translation vector
    :param vector:
    :param scaling:
    :return: Scaled translation vector
    """
    return vector * scaling

def scale_quat(quat, scaling):
    """
    Scale the rotation of quaternion
    :param quat: Rotation quaternion [w, x, y, z]
    :param scaling: Scaling factor
    :return: Scaled rotation quaternion
    """
    axis, angle = quat2axangle(quat)
    quat = axangle2quat(axis, angle * scaling)
    return quat