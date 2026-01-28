#include <torch/extension.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

torch::Tensor hgemm_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be CUDA");
    TORCH_CHECK(B.is_cuda(), "B must be CUDA");
    TORCH_CHECK(A.scalar_type() == torch::kFloat16, "A must be FP16");
    TORCH_CHECK(B.scalar_type() == torch::kFloat16, "B must be FP16");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    cublasHandle_t handle;
    cublasCreate(&handle);

    half* A_ptr = (half*)A.data_ptr<at::Half>();
    half* B_ptr = (half*)B.data_ptr<at::Half>();
    half* C_ptr = (half*)C.data_ptr<at::Half>();

    float alpha = 1.0f;
    float beta  = 0.0f;

    // ⚠️ cuBLAS 是 column-major
    // C = Bᵀ × Aᵀ  等价于 row-major A × B
    cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B_ptr, CUDA_R_16F, N,
        A_ptr, CUDA_R_16F, K,
        &beta,
        C_ptr, CUDA_R_16F, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );

    cublasDestroy(handle);
    return C;
}
