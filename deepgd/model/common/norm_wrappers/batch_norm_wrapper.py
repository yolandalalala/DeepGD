from .norm_wrapper import NormWrapper

import torch
import torch_geometric as pyg


@NormWrapper.register(norm_type=pyg.nn.BatchNorm)
class BatchNormWrapper(NormWrapper):
    """Surrogate wrapper for pyg.nn.BatchNorm"""

    def forward(self, *,
                node_feat: torch.FloatTensor,
                batch_index: torch.LongTensor) -> torch.FloatTensor:
        """The surrogate forward for underlying BatchNorm layer.
        :param node_feat: node features
        :param batch_index: dummy batch index which is not needed for BatchNorm
        :return: output from BatchNorm
        """
        return self.bn(node_feat)
