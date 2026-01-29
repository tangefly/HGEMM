#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be CUDA")
#define CHECK_FP16(x) TORCH_CHECK(x.scalar_type() == torch::kFloat16, #x " must be FP16")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

__global__ void naive_hgemm_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int K, int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            half a = A[row * K + k];
            half b = B[k * N + col];
            acc += __half2float(a) * __half2float(b);
        }
        C[row * N + col] = __float2half(acc);
    }
}

torch::Tensor naive_hgemm(torch::Tensor A, torch::Tensor B) {
    CHECK_CUDA(A);
    CHECK_CUDA(B);
    CHECK_FP16(A);
    CHECK_FP16(B);
    CHECK_CONTIGUOUS(A);
    CHECK_CONTIGUOUS(B);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    TORCH_CHECK(B.size(0) == K, "B.shape[0] must equal A.shape[1]");

    auto C = torch::zeros({M, N}, A.options());

    const half* A_ptr = reinterpret_cast<half*>(A.data_ptr<at::Half>());
    const half* B_ptr = reinterpret_cast<half*>(B.data_ptr<at::Half>());
    half* C_ptr = reinterpret_cast<half*>(C.data_ptr<at::Half>());

    dim3 block(16, 16);
    dim3 grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y
    );

    naive_hgemm_kernel<<<grid, block>>>(
        A_ptr, B_ptr, C_ptr,
        M, K, N
    );

    return C;
}
