from typing import Dict, List

import numpy as np

from src.control.utils.arm_state import ArmState

class OSAction:
    """
    The OSAction (operational space action) class holds a dictionary of arm states
    """
    def __init__(self, state_dict : Dict[str, ArmState]):
        assert isinstance(state_dict, Dict), "state_dict must be a dictionary"
        assert all([isinstance(state_dict[key], ArmState) for key in state_dict]), "All values in state_dict must be ArmState objects"
        self.state_dict = state_dict

    def __getitem__(self, __name: str) -> ArmState:
        return self.state_dict[__name]

    def __setitem__(self, __name: str, __value: ArmState) -> None:
        self.state_dict[__name] = __value

    def keys(self) -> List[str]:
        return list(self.state_dict.keys())

    def values(self) -> List[ArmState]:
        return list(self.state_dict.values())

    def items(self):
        return self.state_dict.items()

    def __str__(self):
        return str(self.state_dict)

    def __len__(self):
        return len(self.state_dict)

    def get(self):
        return self.state_dict

    def flatten(self) -> np.ndarray:
        return np.concatenate([v.flatten() for v in self.values()], axis=1)

    @classmethod
    def from_flattened(cls, array : np.ndarray, device_names : List[str]):
        # euler angles or quaternions
        if len(array) == len(device_names) * (3 + 3 + 1):
            rot = "euler"
        elif len(device_names) * (3 + 4 + 1):
            rot = "quat"
        else:
            raise ValueError

        state_dict = {}
        start_idx = 0
        for name in device_names:
            end_idx = start_idx + (7 if rot == "euler" else 8)
            state_dict[name] = ArmState.from_flattened(array[start_idx:end_idx])
            start_idx = end_idx

        return cls(state_dict)

