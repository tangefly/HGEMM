import torch
import pytest
import hgemm_ext

def test_thread_tile_hgemm_basic():
    M = K = N = 4096
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)

    C = hgemm_ext.thread_tile_hgemm(A, B)

    C_ref = A @ B

    # 数值误差允许
    assert torch.allclose(C, C_ref, rtol=1e-2, atol=1e-2)
