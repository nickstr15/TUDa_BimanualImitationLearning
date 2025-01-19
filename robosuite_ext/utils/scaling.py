from robosuite_ext.utils.transform_utils import quat2axisangle, axisangle2quat


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
    :param quat: Rotation quaternion [x, y, z, w]
    :param scaling: Scaling factor
    :return: Scaled rotation quaternion
    """
    aa = quat2axisangle(quat) * scaling
    return axisangle2quat(aa)