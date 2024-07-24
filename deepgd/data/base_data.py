from .transforms import BatchAppendColumn

from abc import ABC, abstractmethod
from typing import Mapping, Optional, Union, Iterable, Any
from typing_extensions import Self

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data, Batch


class BaseData(Data, ABC):

    @classmethod
    def get_optional_fields(cls):
        if not hasattr(cls, "_optional_fields"):
            cls._optional_fields = []
        return cls._optional_fields

    @classmethod
    def set_optional_fields(cls, fields: Optional[list] = None):
        cls._optional_fields = [] if fields is None else fields

    @classmethod
    @abstractmethod
    def field_annotations(cls) -> dict[str, type]:
        raise NotImplementedError

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if "metaindex" in key:
            return self.perm_index.shape[1]
        return super().__inc__(key, value, *args, **kwargs)

    def __getattr__(self, key: str) -> Any:
        if key not in self and key in self.field_annotations():
            for suffix in ["index", "attr", "weight"]:
                if key.endswith("_" + suffix):
                    metaindex_key = key.replace(suffix, "metaindex")
                    if metaindex_key not in self.field_annotations():
                        continue
                    indexer = getattr(self, metaindex_key)
                    if indexer is None:
                        return None
                    if suffix == "index":
                        indexer = slice(None), indexer
                    return getattr(self, f"perm_{suffix}")[indexer]
            else:
                return None
        return super().__getattr__(key)

    def __getattribute__(self, key):
        value = object.__getattribute__(self, key)
        if isinstance(value, type(self).Field) or value is NotImplemented:
            value = self.__getattr__(key)
        return value

    # noinspection PyPep8Naming
    @classmethod
    def new(cls, G: nx.Graph) -> Optional[Self]:
        data = cls(G=G)
        if data.pre_filter():
            return data.pre_transform().static_transform().dynamic_transform().post_transform()
        return None

    @abstractmethod
    def pre_filter(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def pre_transform(self) -> Self:
        raise NotImplementedError

    @abstractmethod
    def static_transform(self) -> Self:
        raise NotImplementedError

    @abstractmethod
    def dynamic_transform(self) -> Self:
        raise NotImplementedError

    @abstractmethod
    def post_transform(self) -> Self:
        raise NotImplementedError

    def append(self, tensor: torch.Tensor, *, name: str, like: Optional[str] = None):
        if like is not None:
            self[name] = tensor.to(self[like])
        else:
            self[name] = tensor
        if isinstance(self, Batch):
            return BatchAppendColumn(
                attr_name=name,
                like=like,
                dtype=tensor.dtype if like is None else None,
                device=tensor.device if like is None else None
            )(self)
        return self
