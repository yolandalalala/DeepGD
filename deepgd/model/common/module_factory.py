from typing import Callable, Optional

import torch
from torch import nn
import torch_geometric as pyg


class ModuleFactory:

    @property
    def identity(self) -> Callable[..., nn.Identity]:
        return nn.Identity

    @property
    def linear(self) -> Callable[..., nn.Linear]:
        return nn.Linear

    @property
    def relu(self) -> Callable[..., nn.ReLU]:
        return nn.ReLU

    @property
    def leaky_relu(self) -> Callable[..., nn.LeakyReLU]:
        return nn.LeakyReLU

    @property
    def tanh(self) -> Callable[..., nn.Tanh]:
        return nn.Tanh

    @property
    def sigmoid(self) -> Callable[..., nn.Sigmoid]:
        return nn.Sigmoid

    @property
    def batch_norm(self) -> Callable[..., nn.BatchNorm1d]:
        return nn.BatchNorm1d

    @property
    def pyg_batch_norm(self) -> Callable[..., pyg.nn.BatchNorm]:
        return pyg.nn.BatchNorm

    @property
    def pyg_graph_norm(self) -> Callable[..., pyg.nn.BatchNorm]:
        return pyg.nn.GraphNorm

    # TODO: @ModuleFactory.register

    def __init__(self, name: Optional[str]):
        self.name = name or "identity"

    def __call__(self, *args, **kwargs) -> nn.Module:
        return getattr(self, self.name)(*args, **kwargs)
