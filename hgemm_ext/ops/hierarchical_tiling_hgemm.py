import torch
from .. import _C

def hierarchical_tiling_hgemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _C.hierarchical_tiling_hgemm(a, b)
