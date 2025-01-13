import collections

import torch
import torch.nn as nn

from robosuite_ext.imitation_learning.models.model import ModelBase
from robosuite_ext.imitation_learning.core.helpers import get_activation_fn
from robosuite_ext.imitation_learning.models.normalizer import LinearNormalizer


class MLP(ModelBase):
    def __init__(self, input_dim, output_dim, hidden_sizes=None, activation="relu", output_activation=None):
        super().__init__(input_dim, output_dim)

        if hidden_sizes is None:
            hidden_sizes = [256, 256]

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.hidden_sizes = hidden_sizes

        self.activation = get_activation_fn(activation)
        self.output_activation = get_activation_fn(output_activation) if output_activation is not None else None

        self.layers = nn.ModuleList()

        # input layer with first activation function
        self.layers.append(nn.Linear(input_dim, hidden_sizes[0]))
        self.layers.append(self.activation())

        # hidden layers with activation function
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.layers.append(self.activation())

        # output layer with output activation function
        self.layers.append(nn.Linear(hidden_sizes[-1], output_dim))
        if output_activation is not None:
            self.layers.append(self.output_activation())

        self.model = nn.Sequential(*self.layers)

    def forward(self, x) -> torch.Tensor:
        assert isinstance(x, collections.OrderedDict), \
            f"Input must be ordered dict, got {type(x)}"

        #convert ordered dictionary {"key", (B, *)} to single tensor with batch size (B, *)
        if len(list(x.values())[0].shape) == 1:
            x = torch.cat(list(x.values()))
        else:
            x = torch.cat(list(x.values()), dim=-1)

        # flatten the input
        if len(x.shape) == 1:
            x = torch.unsqueeze(x, dim=0)
        else:
            x = x.view(x.size(0), -1)

        return self.model.forward(x)