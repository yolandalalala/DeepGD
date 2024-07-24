from typing import Optional, Union

import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Batch


class BatchAppendColumn(BaseTransform):

    def __init__(self,
                 attr_name: str,
                 like: Optional[str] = None,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[Union[str, torch.device]] = None):
        super().__init__()
        self.attr_name = attr_name
        self.like = like
        self.dtype = dtype
        self.device = device

    def __call__(self, batch: Batch) -> Batch:
        tensor = batch[self.attr_name]
        data_list = batch.to_data_list()
        like_name = self.like
        if like_name is None:
            for col, _ in data_list[0]:
                target = batch[col]
                if isinstance(target, torch.Tensor) and target.shape == tensor.shape:
                    like_name = col
                    break
        shape = None
        if like_name is None:
            shape = tensor.shape
            shape[0] /= len(batch)
        for data in data_list:
            data[self.attr_name] = torch.zeros(*shape or data[like_name].shape)
        batch = Batch.from_data_list(data_list)
        assert batch[self.attr_name].shape == tensor.shape
        tensor = tensor.to(batch[like_name] if like_name is not None else tensor)
        tensor = tensor.to(self.dtype if self.dtype is not None else tensor)
        tensor = tensor.to(self.device if self.device is not None else tensor)
        batch[self.attr_name] = tensor
        return batch
