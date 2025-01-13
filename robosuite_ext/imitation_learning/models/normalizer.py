import collections
from typing import Union, Dict

import torch
from torch import nn

from robosuite_ext.imitation_learning.core.tensor_dict import StaticTensorDict


class LinearNormalizer(StaticTensorDict):

    @torch.no_grad()
    def fit(
        self,
        data: Union[Dict[str, torch.Tensor], collections.OrderedDict[str, torch.Tensor], torch.Tensor],
        dtype: torch.dtype = torch.float32,
        output_min: float = -1.0,
        output_max: float = 1.0,
        range_eps: float = 1e-4,
        fit_offset: bool = True
    ) -> None:
        """
        Fit the normalizer to the data.
        :param data: dictionary of data or tensor of data, shape {"key" : (B, *) } or (B, *)
        :param dtype: desired data type
        :param output_min: minimum value for the output after normalization
        :param output_max: maximum value for the output after normalization
        :param range_eps: epsilon, if the range is smaller than this value, the range is set to output_max - output_min
        :param fit_offset: whether to fit the offset, set to false if the data is already centered
        """

        if isinstance(data, dict) or isinstance(data, collections.OrderedDict):
            for key, value in data.items():
                self.params_dict[key] = self._fit_single(
                    value, dtype, output_min, output_max, range_eps, fit_offset
                )

        elif isinstance(data, torch.Tensor):
            self.params_dict["_"] = self._fit_single(
                data, dtype, output_min, output_max, range_eps, fit_offset
            )

        else:
            raise ValueError("Data must be a dictionary, ordered dictionary or tensor.")

    def __call__(
        self,
        x: Union[Dict[str, torch.Tensor], collections.OrderedDict[str, torch.Tensor], torch.Tensor]
    ) -> Union[Dict[str, torch.Tensor], collections.OrderedDict[str, torch.Tensor], torch.Tensor]:
        """
        Normalize the input.
        :param x: dictionary of data or tensor of data, shape {"key" : (B, *) } or (B, *)
        :return: normalized data
        """
        return self.normalize(x)

    def normalize(
        self,
            x: Union[Dict[str, torch.Tensor], collections.OrderedDict[str, torch.Tensor], torch.Tensor]
    ) -> Union[Dict[str, torch.Tensor], collections.OrderedDict[str, torch.Tensor], torch.Tensor]:
        """
        Normalize the input.
        :param x: dictionary of data or tensor of data, shape {"key" : (B, *) } or (B, *)
        :return: normalized data
        """
        return self._normalize(x, forward=True)

    def denormalize(
        self,
        x: Union[Dict[str, torch.Tensor], collections.OrderedDict[str, torch.Tensor], torch.Tensor]
    ) -> Union[Dict[str, torch.Tensor], collections.OrderedDict[str, torch.Tensor], torch.Tensor]:
        """
        Denormalize the input.
        :param x: dictionary of data or tensor of data, shape {"key" : (B, *) } or (B, *)
        :return: unnormalized data
        """
        return self._normalize(x, forward=False)

    def _normalize(
        self,
        x: Union[Dict[str, torch.Tensor], collections.OrderedDict[str, torch.Tensor], torch.Tensor],
        forward: bool = True
    ) -> Union[Dict[str, torch.Tensor], collections.OrderedDict[str, torch.Tensor], torch.Tensor]:
        """
        Normalize the input.
        :param x: dictionary of data or tensor of data, shape {"key" : (B, *) } or (B, *)
        :param forward: whether to normalize or denormalize
        :return: normalized data
        """
        if isinstance(x, dict) or isinstance(x, collections.OrderedDict):
            return collections.OrderedDict({
                key: self._normalize_single(value, key, forward) for key, value in x.items()
            })

        elif isinstance(x, torch.Tensor):
            return self._normalize_single(x, "_", forward)

        else:
            raise ValueError("Data must be a dictionary, ordered dictionary or tensor.")

    def _normalize_single(self, x: torch.Tensor, key: str = "_", forward: bool = True) -> torch.Tensor:
        """
        Normalize the input.
        :param x: input data of shape (B, *)
        :param key: key of the data, used to get the parameters
        :param forward: whether to normalize or denormalize
        :return: normalized data
        """
        params = self.params_dict.get(key, None)
        if params is None:
            raise ValueError(f"Key {key} not found in the normalizer.")

        scale = params["scale"]
        offset = params["offset"]

        # assert that the dimensions match, x must have (B, *), scale and offset must have (*,)
        assert x.dim() > 1, "Input must have at least 2 dimensions."
        assert x.size()[1:] == scale.size(), "Input and scale dimensions must match, got {} and {}.".format(
            x.size()[1:], scale.size()
        )

        if forward:
            return x * scale + offset
        else:
            return (x - offset) / scale

    @staticmethod
    def _fit_single(
        x: torch.Tensor,
        dtype: torch.dtype = torch.float32,
        output_min: float = -1.0,
        output_max: float = 1.0,
        range_eps: float = 1e-4,
        fit_offset: bool = True
    ) -> nn.ParameterDict:
        """
        Fit the normalizer to the data.
        :param x: input data of shape (B, *)
        :param dtype: desired data type
        :param output_min: minimum value for the output after normalization
        :param output_max: maximum value for the output after normalization
        :param range_eps: epsilon, if the range is smaller than this value, the range is set to output_max - output_min
        :param fit_offset: whether to fit the offset, set to false if the data is already centered
        :return:
        """

        # check that x has batch dimension
        assert x.dim() > 1, "Input must have at least 2 dimensions."

        # store source shape
        src_shape = x.size()
        # flatten the input
        x = x.view(x.size(0), -1)

        # compute input statistics
        x_min = x.min(dim=0, keepdim=True)[0]
        x_max = x.max(dim=0, keepdim=True)[0]

        # compute scale and offset
        if fit_offset:
            x_range = x_max - x_min
            ignore_dim = x_range < range_eps
            x_range[ignore_dim] = output_min - output_max
            scale = (output_max - output_min) / x_range
            offset = output_min - x_min * scale
            offset[ignore_dim] = (output_max + output_min) / 2 - x_min[ignore_dim]
        else: # assume centered data
            assert output_max > 0
            assert output_min < 0
            # unit abs
            output_abs = min(abs(output_min), abs(output_max))
            x_abs = torch.maximum(torch.abs(x_min), torch.abs(x_max))
            ignore_dim = x_abs < range_eps
            x_abs[ignore_dim] = output_abs # don't scale constant channels
            scale = output_abs / x_abs
            offset = torch.zeros_like(x_min)

        # reshape scale and offset
        scale = scale.view(src_shape[1:])
        offset = offset.view(src_shape[1:])

        # create parameter dictionary
        params_dict = nn.ParameterDict({
            "scale": nn.Parameter(scale.to(dtype)),
            "offset": nn.Parameter(offset.to(dtype))
        })

        # disable gradients
        for p in params_dict.values():
            p.requires_grad_(False)

        return params_dict



