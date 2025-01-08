from abc import ABC

import torch
import torch.nn as nn

class StaticTensorDict(ABC, nn.Module):
    def __init__(self, params_dict: nn.ParameterDict = None):
        super().__init__()
        self.params_dict = params_dict if params_dict is not None else nn.ParameterDict()
        self.params_dict.requires_grad_(False)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        def dfs_add(dest, keys, value: torch.Tensor):
            if len(keys) == 1:
                dest[keys[0]] = value
                return

            if keys[0] not in dest:
                dest[keys[0]] = nn.ParameterDict()
            dfs_add(dest[keys[0]], keys[1:], value)

        def load_dict(_state_dict, _prefix):
            out_dict = nn.ParameterDict()
            for key, value in _state_dict.items():
                value: torch.Tensor
                if key.startswith(_prefix):
                    param_keys = key[len(_prefix):].split('.')[1:]
                    # if len(param_keys) == 0:
                    #     import pdb; pdb.set_trace()
                    dfs_add(out_dict, param_keys, value.clone())
            return out_dict

        self.params_dict = load_dict(state_dict, prefix + 'params_dict')
        self.params_dict.requires_grad_(False)

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype