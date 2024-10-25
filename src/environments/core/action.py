from typing import Dict, List

import numpy as np

from src.control.utils.ee_state import EEState

class OSAction:
    """
    The OSAction (operational space action) class holds a dictionary of arm states
    """
    def __init__(self, state_dict : Dict[str, EEState]):
        assert isinstance(state_dict, Dict), "state_dict must be a dictionary"
        assert all([isinstance(state_dict[key], EEState) for key in state_dict]), "All values in state_dict must be ArmState objects"
        self.state_dict = state_dict

    def __getitem__(self, __name: str) -> EEState:
        return self.state_dict[__name]

    def __setitem__(self, __name: str, __value: EEState) -> None:
        self.state_dict[__name] = __value

    def keys(self) -> List[str]:
        return list(self.state_dict.keys())

    def values(self) -> List[EEState]:
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
        return np.concatenate([v.flatten() for v in self.values()])

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
            state_dict[name] = EEState.from_flattened(array[start_idx:end_idx])
            start_idx = end_idx

        return cls(state_dict)

