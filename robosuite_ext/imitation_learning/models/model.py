from abc import ABC

import torch
from torch import nn

from robosuite_ext.imitation_learning.models.normalizer import LinearNormalizer


class ModelBase(ABC, nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype