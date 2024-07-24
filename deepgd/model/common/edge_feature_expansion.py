from deepgd.constants import EPS
from .traced_module import TracedModule

from typing import Callable, Any, Optional
from dataclasses import dataclass, asdict
from functools import wraps
from attrs import define, frozen

import torch
from torch import nn, jit


@define(kw_only=True, eq=False, repr=False, slots=False)
class EdgeFeatureExpansion(nn.Module):

    @jit.script
    @dataclass(kw_only=True)
    class Config:
        node_feat_dim: int
        edge_attr_dim: int

    @dataclass(kw_only=True)
    class Expansions:
        src_feat: bool = False
        dst_feat: bool = False
        diff_vec: bool = False
        unit_vec: bool = False
        vec_norm: bool = False
        vec_norm_inv: bool = False
        vec_norm_square: bool = False
        vec_norm_inv_square: bool = False
        edge_attr_inv: bool = False
        edge_attr_square: bool = False
        edge_attr_inv_square: bool = False

    @staticmethod
    def feature_expansion(func: Callable[[Any, dict[str, torch.Tensor]], torch.Tensor]):
        @wraps(func)
        def decorator(self, v: dict[str, torch.Tensor]):
            if func.__name__ not in v:
                v[func.__name__] = func(self, v)
            return v[func.__name__]
        return decorator

    @staticmethod
    def _get_src_feat_dst_feat(v: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        node_feat = v["node_feat"]
        edge_index = v["edge_index"]
        src_dst_feat = node_feat[edge_index]
        src_feat = src_dst_feat[1, ...]
        dst_feat = src_dst_feat[0, ...]
        v["src_feat"] = src_feat
        v["dst_feat"] = dst_feat
        return src_feat, dst_feat

    @feature_expansion
    def src_feat(self, v: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._get_src_feat_dst_feat(v)[0]

    @feature_expansion
    def dst_feat(self, v: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._get_src_feat_dst_feat(v)[1]

    @feature_expansion
    def diff_vec(self, v: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.src_feat(v) - self.dst_feat(v)

    @feature_expansion
    def unit_vec(self, v: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.diff_vec(v) / (self.vec_norm(v) + self.eps)

    @feature_expansion
    def vec_norm(self, v: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.diff_vec(v).norm(dim=1, p=2, keepdim=True)

    @feature_expansion
    def vec_norm_inv(self, v: dict[str, torch.Tensor]) -> torch.Tensor:
        return 1 / (self.vec_norm(v) + self.eps)

    @feature_expansion
    def vec_norm_square(self, v: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.vec_norm(v) ** 2

    @feature_expansion
    def vec_norm_inv_square(self, v: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.vec_norm_inv(v) ** 2

    @feature_expansion
    def edge_attr_inv(self, v: dict[str, torch.Tensor]) -> torch.Tensor:
        return 1 / (v["edge_attr"] + self.eps)

    @feature_expansion
    def edge_attr_square(self, v: dict[str, torch.Tensor]) -> torch.Tensor:
        return v["edge_attr"] ** 2

    @feature_expansion
    def edge_attr_inv_square(self, v: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.edge_attr_inv(v) ** 2

    @staticmethod
    def _get_example_inputs(*,
                            node_feat_dim: int = 1,
                            edge_attr_dim: int = 1,
                            n: int = 1,
                            m: int = 1):
        return dict(
            node_feat=torch.zeros(n, node_feat_dim).float(),
            edge_attr=torch.zeros(m, edge_attr_dim).float(),
            edge_index=torch.zeros(2, m).long()
        )

    def get_feature_channels(self, *,
                             node_feat_dim: Optional[int] = None,
                             edge_attr_dim: Optional[int] = None):
        return self(**self._get_example_inputs(
            node_feat_dim=node_feat_dim or self.config.node_feat_dim or 0,
            edge_attr_dim=edge_attr_dim or self.config.edge_attr_dim or 0
        )).shape[-1]

    config: Config = Config(
        node_feat_dim=-1,
        edge_attr_dim=-1
    )
    expansions: Expansions = Expansions()
    eps: float = EPS

    def __attrs_post_init__(self):
        super().__init__()

        # TODO: make it a method with @traced
        def get_edge_feat(node_feat: torch.Tensor,
                          edge_attr: torch.Tensor,
                          edge_index: torch.Tensor):
            feat_list = [edge_attr]
            for feat, include in asdict(self.expansions).items():
                if include:
                    feat_list.append(getattr(self, feat)(dict(
                        node_feat=node_feat,
                        edge_attr=edge_attr,
                        edge_index=edge_index
                    )))
            return torch.cat(feat_list, dim=-1)

        # TODO: self.get_edge_feat.trace(example_inputs=self._get_example_inputs())
        self.get_edge_feat: TracedModule = TracedModule(
            func=get_edge_feat,
            example_inputs=self._get_example_inputs()
        )

    def forward(self, *,
                node_feat: torch.FloatTensor,
                edge_attr: torch.FloatTensor,
                edge_index: torch.LongTensor) -> torch.FloatTensor:
        if self.config.node_feat_dim > 0:
            assert node_feat.shape[-1] == self.config.node_feat_dim
        if self.config.edge_attr_dim > 0:
            assert edge_attr.shape[-1] == self.config.edge_attr_dim
        return self.get_edge_feat(dict(
            node_feat=node_feat,
            edge_attr=edge_attr,
            edge_index=edge_index
        ))


EdgeFeatureExpansion.__annotations__ = {
    'config': EdgeFeatureExpansion.Config
}
