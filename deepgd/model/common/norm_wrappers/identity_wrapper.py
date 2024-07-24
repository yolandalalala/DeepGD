from .norm_wrapper import NormWrapper

import torch
from torch import nn


@NormWrapper.register(norm_type=nn.Identity)
class IdentityWrapper(NormWrapper):
    """Surrogate wrapper for torch.nn.Identity"""

    def forward(self, *,
                node_feat: torch.FloatTensor,
                batch_index: torch.LongTensor) -> torch.FloatTensor:
        """The surrogate forward for underlying Identity layer.
        :param node_feat: node features
        :param batch_index: dummy batch index which is not needed for Identity
        :return: output from Identity
        """
        return self.bn(node_feat)
