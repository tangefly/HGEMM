import torch
import hgemm_ext

def profile_naive_hgemm():
    M = K = N = 4096
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)

    torch.cuda.synchronize()
    C = hgemm_ext.naive_hgemm(A, B)
    torch.cuda.synchronize()

if __name__ == "__main__":
    profile_naive_hgemm()
