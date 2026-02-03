import torch
from .. import _C

def thread_tile_hgemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _C.thread_tile_hgemm(a, b)
