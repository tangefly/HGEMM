import torch
import hgemm_ext

def profile_hierarchical_tiling_hgemm():
    M = K = N = 4096
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)

    torch.cuda.synchronize()
    C = hgemm_ext.hierarchical_tiling_hgemm(A, B)
    torch.cuda.synchronize()

if __name__ == "__main__":
    profile_hierarchical_tiling_hgemm()
