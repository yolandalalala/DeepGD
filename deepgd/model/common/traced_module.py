from typing import Callable, Any

import torch
from torch import nn, jit


class TracedModule(nn.Module):

    def __new__(cls, func: Callable, example_inputs: dict[str, torch.Tensor]) -> jit.ScriptModule:
        module = super().__new__(cls)
        module.__init__(func=func, example_inputs=example_inputs)
        return jit.trace_module(mod=module, inputs=dict(forward=example_inputs))

    def __init__(self, func: Callable, example_inputs: dict[str, torch.Tensor]):
        super().__init__()
        self.func: Callable = func

    # TODO: dynamically generate forward method
    def forward(self, kwargs: dict, /) -> Any:
        return self.func(**kwargs)
