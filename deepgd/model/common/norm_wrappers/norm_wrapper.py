from typing_extensions import Self

import torch
from torch import nn


class NormWrapper(nn.Module):
    """This is a surrogate wrapper around pyg normalization layers used to make
    sure the interface is static across different kinds of normalizations, so
    that the model can be complied into static TorchScript even if the model
    components are dynamic.
    """

    _registry: dict[type[nn.Module], type[Self]] = {}

    @classmethod
    def register(cls, norm_type):
        def decorator(wrapper_type):
            cls._registry[norm_type] = wrapper_type
            return wrapper_type
        return decorator

    def __new__(cls, bn: nn.Module) -> nn.Module:
        for norm_type, wrapper_type in cls._registry.items():
            if isinstance(bn, norm_type):
                return super().__new__(wrapper_type)
        return super().__new__(cls)

    def __init__(self, bn: nn.Module):
        super().__init__()
        self.bn: nn.Module = bn

    def forward(self, *,
                node_feat: torch.FloatTensor,
                batch_index: torch.LongTensor) -> torch.FloatTensor:
        """
        :param node_feat: node features
        :param batch_index: batch index
        :return: normalized output
        """
        return self.bn(node_feat, batch=batch_index)
