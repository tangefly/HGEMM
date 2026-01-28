import torch
from .. import _C

def hgemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _C.hgemm(a, b)
