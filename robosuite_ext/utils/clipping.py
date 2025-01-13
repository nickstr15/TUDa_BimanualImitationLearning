import numpy as np
from robosuite_ext.utils.transform_utils import quat2axisangle, axisangle2quat


def clip_translation(v: np.ndarray, v_min: np.ndarray, v_max: np.ndarray) -> np.ndarray:
    """
    Clip the translation vector.
    :param v: Translation vector
    :param v_min: Min values per axis
    :param v_max: Max values per axis
    :return: Clipped translation vector
    """
    return np.clip(v, v_min, v_max)

def clip_quat_by_axisangle(q: np.ndarray, aa_min: np.ndarray, aa_max: np.ndarray) -> np.ndarray:
    """
    Clip the rotation quaternion to an euler range.
    :param q: Rotation quaternion [x, y, z, w]
    :param aa_min: Min axisangle per axis in radians
    :param aa_max: Max axisangle per axis in radians
    :return: Clipped rotation quaternion
    """
    aa = quat2axisangle(q)
    aa_clipped = np.clip(aa, aa_min, aa_max)
    return axisangle2quat(aa_clipped)
