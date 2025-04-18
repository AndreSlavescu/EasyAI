#include <iostream>
#include <vector>
#include <random>
#include <cmath>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "../cuda_tiling_template_utils.cuh"

constexpr int TILE_M = 16;
constexpr int TILE_N = 16;
constexpr int TILE_K = 16;

template<int TM, int TN, int TK>
__global__ void tiledSgemm(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* C,
    int M,
    int N,
    int K
) {
    __shared__ float A_tile[TM][TK + 1];
    __shared__ float B_tile[TK][TN + 1];
    
    unsigned int row = blockIdx.y * TM + threadIdx.y;
    unsigned int col = blockIdx.x * TN + threadIdx.x;

    float acc = 0.0f;

    for (unsigned int k0 = 0; k0 < K; k0 += TK) {
        int A_row_origin = blockIdx.y * TILE_M; 
        int A_col_origin = k0; 
        tileLoadFromGlobal2D(
            A_tile,
            A,
            /* ld_gmem = */ K, // A has shape MxK, row-major
            A_row_origin,
            A_col_origin,
            /* rowBound = */ M,
            /* colBound = */ K
        );

        int B_row_origin = k0;
        int B_col_origin = blockIdx.x * TILE_N;
        tileLoadFromGlobal2D(
            B_tile,
            B,
            /* ld_gmem = */ N, // B has shape KxN, row-major
            B_row_origin,
            B_col_origin,
            /* rowBound = */ K,
            /* colBound = */ N
        );

        if (row < M && col < N) {
            #pragma unroll
            for (unsigned int k_tile_idx = 0; k_tile_idx < TK && k0 + k_tile_idx < K; ++k_tile_idx) {
                acc += A_tile[threadIdx.y][k_tile_idx] * B_tile[k_tile_idx][threadIdx.x];
            }
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

void host_reference_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
                acc += A[i * K + k] * B[k * N + j];
            C[i * N + j] = acc;
        }
}

float compute_l2_error(const float* ref, const float* test, int size) {
    float sum_sq = 0.0f;
    float ref_sq = 0.0f;
    for (int i = 0; i < size; ++i) {
        float diff = ref[i] - test[i];
        sum_sq += diff * diff;
        ref_sq += ref[i] * ref[i];
    }
    return std::sqrt(sum_sq / (ref_sq + 1e-6f));
}

int main() {
    constexpr int M = 128;
    constexpr int N = 128;
    constexpr int K = 128;

    std::vector<float> A(M * K), B(K * N), C_ref(M * N), C_gpu(M * N), C_blas(M * N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    for (auto& x : A) x = dist(rng);
    for (auto& x : B) x = dist(rng);

    float *d_A, *d_B, *d_C;
    CUDA_ASSERT(cudaMalloc(&d_A, sizeof(float) * A.size()));
    CUDA_ASSERT(cudaMalloc(&d_B, sizeof(float) * B.size()));
    CUDA_ASSERT(cudaMalloc(&d_C, sizeof(float) * C_gpu.size()));

    CUDA_ASSERT(cudaMemcpy(d_A, A.data(), sizeof(float) * A.size(), cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(d_B, B.data(), sizeof(float) * B.size(), cudaMemcpyHostToDevice));

    dim3 blockDim(TILE_N, TILE_M);
    dim3 gridDim(CEIL_DIV(N, TILE_N), CEIL_DIV(M, TILE_M));

    tiledSgemm<TILE_M, TILE_N, TILE_K><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_ASSERT(cudaMemcpy(C_gpu.data(), d_C, sizeof(float) * C_gpu.size(), cudaMemcpyDeviceToHost));

    // CPU reference
    host_reference_gemm(A.data(), B.data(), C_ref.data(), M, N, K);

    // cuBLAS
    cublasHandle_t handle;
    CUBLAS_ASSERT(cublasCreate(&handle));
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_ASSERT(cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, N,
        d_A, K,
        &beta,
        d_C, N));
    CUDA_ASSERT(cudaMemcpy(C_blas.data(), d_C, sizeof(float) * C_blas.size(), cudaMemcpyDeviceToHost));
    CUBLAS_ASSERT(cublasDestroy(handle));

    float err_gpu = compute_l2_error(C_ref.data(), C_gpu.data(), M * N);
    float err_blas = compute_l2_error(C_ref.data(), C_blas.data(), M * N);

    std::cout << "[L2 Error vs CPU]  Tiled GEMM: " << err_gpu << "\n";
    std::cout << "[L2 Error vs CPU]  cuBLAS:     " << err_blas << "\n";

    CUDA_ASSERT(cudaFree(d_A));
    CUDA_ASSERT(cudaFree(d_B));
    CUDA_ASSERT(cudaFree(d_C));
}
