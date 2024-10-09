from typing import Tuple, List

import numpy as np


def range_shift(
        vals: np.ndarray,
        current_range: np.ndarray | List | Tuple,
        target_range: np.ndarray | List | Tuple
) -> np.ndarray:
    """
    Shift the value from the current range to the target range

    :param vals: values to shift
    :param current_range: current range of the value
    :param target_range: target range of the value
    :return: shifted value
    """
    assert len(current_range) == 2 and len(target_range) == 2, "Both ranges must be of length 2"

    return (vals - current_range[0]) / (current_range[1] - current_range[0]) * (target_range[1] - target_range[0]) + \
        target_range[0]
