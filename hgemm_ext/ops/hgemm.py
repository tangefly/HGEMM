import torch
from .. import _C

def hgemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _C.hgemm(a, b)

def hierarchical_tiling_hgemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _C.hierarchical_tiling_hgemm(a, b)

def naive_hgemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _C.naive_hgemm(a, b)

def thread_tile_hgemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _C.thread_tile_hgemm(a, b)

def warp_hgemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _C.warp_hgemm(a, b)