import torch
from .. import _C

def naive_hgemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _C.naive_hgemm(a, b)
