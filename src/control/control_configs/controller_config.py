from typing import Any

class ControllerConfig:
    """
    ControllerConfig class is a wrapper around the controller dictionary
    """
    def __init__(self, ctrlr_dict):
        self.ctrlr_dict = ctrlr_dict

    def __getitem__(self, __name: str) -> Any:
        return self.ctrlr_dict[__name]

    def get_params(self, keys):
        return [self.ctrlr_dict[key] for key in keys]

    def __setitem__(self, __name: str, __value: Any) -> None:
        self.ctrlr_dict[__name] = __value