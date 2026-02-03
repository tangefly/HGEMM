#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be CUDA")
#define CHECK_FP16(x) TORCH_CHECK(x.scalar_type() == torch::kFloat16, #x " must be FP16")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

template<int Bm = 128, int Bn = 128, int Bk = 8, int blockSize = 256, int A_BLOCK_DIM_X = 8,
         int B_BLOCK_DIM_X = 32, int C_BLOCK_DIM_X = 16>
__global__ void thread_tile_hgemm_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int K, int N
) {
    // allocate shared memory for block
    __shared__ half As[Bm][Bk];
    __shared__ half Bs[Bk][Bn];

    // Compute the row and column coordinates 
    // of the top-left element of the tileC handled by the block.
    int tileC_y0 = blockIdx.y * Bm;
    int tileC_x0 = blockIdx.x * Bn;

    // The index of the current thread, using a one-dimensional block configuration.
    int tid = threadIdx.x;

    constexpr int A_BLOCK_DIM_Y = blockSize / A_BLOCK_DIM_X;
    int a_tid_y = tid / A_BLOCK_DIM_X;
    int a_tid_x = tid % A_BLOCK_DIM_X;

    constexpr int B_BLOCK_DIM_Y = blockSize / B_BLOCK_DIM_X;
    int b_tid_y = tid / B_BLOCK_DIM_X;
    int b_tid_x = tid % B_BLOCK_DIM_X;

    constexpr int C_BLOCK_DIM_Y = blockSize / C_BLOCK_DIM_X;
    int c_tid_y = tid / C_BLOCK_DIM_X;
    int c_tid_x = tid % C_BLOCK_DIM_X;

    // Each thread is responsible for computing Tm × Tn elements.
    constexpr int Tm = Bm / C_BLOCK_DIM_Y;
    constexpr int Tn = Bn / C_BLOCK_DIM_X;
    float Ct[Tm][Tn] = {0.0f};

    half regA[Tm] = {__float2half(0.0f)};
    half regB[Tn] = {__float2half(0.0f)};

    // K- Loop
    for(int k = 0; k < K; k += Bk){
        // Copy GlobalMemory To SharedMemory
#pragma unroll
        for(int i = a_tid_y; i < Bm; i += A_BLOCK_DIM_Y){
            int ty = tileC_y0 + i;
#pragma unroll
            for(int j = a_tid_x; j < Bk; j += A_BLOCK_DIM_X){
                int tx = k + j;
                As[i][j] = (ty < M && tx < K) ? A[ty * K + tx] : __float2half(0.0f);
            }
        }

#pragma unroll
        for(int i = b_tid_y; i < Bk; i += B_BLOCK_DIM_Y){
            int ty = k + i;
#pragma unroll
            for(int j = b_tid_x; j < Bn; j += B_BLOCK_DIM_X){
                int tx = tileC_x0 + j;
                Bs[i][j] = (ty < K && tx < N) ? B[ty * N + tx] : __float2half(0.0f);
            }
        }

        __syncthreads();

        // tileA * tileB
#pragma unroll
        for(int p = 0; p < Bk; ++p){
#pragma unroll
            for (int i = 0; i < Tm; ++i) {
                int r = c_tid_y + i * C_BLOCK_DIM_Y;
                regA[i] = As[r][p];
            }
#pragma unroll
            for (int i = 0; i < Tn; ++i) {
                int c = c_tid_x + i * C_BLOCK_DIM_X;
                regB[i] = Bs[p][c];
            }

            for(int i = 0; i < Tm; ++i){
#pragma unroll
                for(int j = 0; j < Tn; ++j){
                    Ct[i][j] += __half2float(regA[i]) * __half2float(regB[j]);
                }
            }
        }
        __syncthreads();
    }

    for(int i = 0; i < Tm; ++i){
        int ty = tileC_y0 + c_tid_y + i * C_BLOCK_DIM_Y;
        for(int j = 0; j < Tn; ++j){
            int tx = tileC_x0 + c_tid_x + j * C_BLOCK_DIM_X;
            if (ty < M && tx < N) { C[ty * N + tx] = __float2half(Ct[i][j]); }
        }
    }
}

torch::Tensor thread_tile_hgemm(torch::Tensor A, torch::Tensor B) {
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

    const int Bm = 128, Bn = 128, Bk = 8, blockSize = 256;
    const int A_BLOCK_DIM_X = 8, B_BLOCK_DIM_X = 32, C_BLOCK_DIM_X = 16;
    dim3 block(blockSize);
    dim3 grid(
        (N + Bn - 1) / Bn,
        (M + Bm - 1) / Bm
    );

    thread_tile_hgemm_kernel<
        Bm, Bn, Bk, blockSize, A_BLOCK_DIM_X, B_BLOCK_DIM_X, C_BLOCK_DIM_X
    ><<<grid, block>>>(
        A_ptr, B_ptr, C_ptr,
        M, K, N
    );

    return C;
}
