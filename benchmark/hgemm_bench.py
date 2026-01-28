import torch
import time
import hgemm_ext

M = N = K = 4096
A = torch.randn(M, K, device="cuda", dtype=torch.float16)
B = torch.randn(K, N, device="cuda", dtype=torch.float16)

# warmup
for _ in range(10):
    hgemm_ext.hgemm(A, B)
    torch.matmul(A, B)

torch.cuda.synchronize()

# custom HGEMM
t0 = time.time()
for _ in range(50):
    hgemm_ext.hgemm(A, B)
torch.cuda.synchronize()
t1 = time.time()

# torch matmul (cuBLAS)
t2 = time.time()
for _ in range(50):
    torch.matmul(A, B)
torch.cuda.synchronize()
t3 = time.time()

print("Custom HGEMM:", t1 - t0)
print("Torch matmul:", t3 - t2)
