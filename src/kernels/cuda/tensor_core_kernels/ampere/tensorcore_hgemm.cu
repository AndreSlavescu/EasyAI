#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "../../cuda_tensor_core_utils.cuh"
#include "../../cuda_utils.h"

__host__ void gemmCPU(
    half* __restrict__ C, 
    const half* __restrict__ A,
    const half* __restrict__ B,
    int M,
    int N,
    int K
) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            // fp32 accumulation for numerical stability
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += static_cast<float>(A[j * K + k]) * static_cast<float>(B[k * N + i]);
            }
            C[i * M + j] = static_cast<half>(acc);
        }
    }
}

// custom vector of size 4 for half types
struct half4 {
    half x;
    half y;
    half z;
    half w;
};

__global__ void gemmTensorCore(
    half* __restrict__ C,
    const half* __restrict__ A,
    const half* __restrict__ B,
    int M,
    int N,
    int K
) {
    /*
    Kernel Flow:

    1. Load tiles into shared memory per threadblock
        - vectorize loads into shared memory
            - pack 4 fp16 values into a half4 vector
            - better for bandwidth, because we write to shared memory in bursts
    2. run wmma_m16n16k16(...) on 4 sub-tiles from the shared memory. 
        - 1024 / (16 * 16) -> 4 sub-tiles.
        - schedule the wmma calls in sequence on each sub-tile.
            - can make use of cuda::pipeline to schedule each wmma call on the seperate sub-tiles
            - each thread within the warp has 8 units of work from the 16x16 sub-tile
    */
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= M && col >= N) return;
    
    // align to 8 bytes for sound vector operations
    //   - 8 bytes is the size of the half4 vector
    __shared__ alignas(sizeof(half) * 4) half A_tile[WARP_SIZE * WARP_SIZE];
    __shared__ alignas(sizeof(half) * 4) half B_tile[WARP_SIZE * WARP_SIZE]; 

    #pragma GCC ivdep 
    for (int i = 0; i < K; i += 4) {
        unsigned int a_idx = row * K + i + threadIdx.x * 4;
        unsigned int b_idx = i * N + col + threadIdx.y * 4;

        // location in shared memory to write to
        unsigned int a_tile_idx = threadIdx.y * WARP_SIZE + threadIdx.x * 4 + i;
        unsigned int b_tile_idx = threadIdx.y * 4 + threadIdx.x * WARP_SIZE + i;

        // load in bursts of 4 addresses at once + handle boundary conditions
        //  build A tile
        if (row < M && (i + threadIdx.x * 4 + 3) < K) {
            half4 A_vec = *reinterpret_cast<half4*>(const_cast<half*>(&A[a_idx]));
            *reinterpret_cast<half4*>(&A_tile[a_tile_idx]) = A_vec;
        } else {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                A_tile[a_tile_idx + j] = (row < M && (i + threadIdx.x * 4 + j) < K) ? 
                                        A[a_idx + j] : (half)0.0f;
            }
        }

        //  build B tile
        if (col < N && (i + threadIdx.y * 4 + 3) < K) {
            half4 B_vec = *reinterpret_cast<half4*>(const_cast<half*>(&B[b_idx]));
            *reinterpret_cast<half4*>(&B_tile[b_tile_idx]) = B_vec;
        } else {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                B_tile[b_tile_idx + j] = (col < N && (i + threadIdx.y * 4 + j) < K) ?
                                        B[(i + threadIdx.y * 4 + j) * N + col] : (half)0.0f;
            }
        }
    }

    threadblock_sync();

    // per-thread accumulation fragment
    float C_frag[8] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };

    // process the 4 sub-tiles (16 * 16 size for each sub-tile)
    #pragma unroll
    for (int sub_tile_idx = 0; sub_tile_idx < 4; sub_tile_idx++) {
        int sub_tile_row = (sub_tile_idx / 2) * 16; // 0 or 16
        int sub_tile_col = (sub_tile_idx % 2) * 16; // 0 or 16

        half* A_subtile_ptr = &A_tile[sub_tile_row * WARP_SIZE + sub_tile_col];
        half* B_subtile_ptr = &B_tile[sub_tile_row * WARP_SIZE + sub_tile_col];
        
        // create matrix descriptors from the subtile address
        uint64_t A_desc = make_smem_desc(A_subtile_ptr);
        uint64_t B_desc = make_smem_desc(B_subtile_ptr);

        // run wmma on the matrix descriptors for both subtiles
        mma_m16n16k16_from_desc<half>(
            C_frag,
            static_cast<uint32_t>(A_desc),
            static_cast<uint32_t>(B_desc)
        );
    }
}

int main() {
    const int N = 1 << 12; // 4096 elements
    
    thrust::host_vector<half> A(N * N);
    thrust::host_vector<half> B(N * N);
    thrust::host_vector<half> C(N * N, 0);
    fillVectorRandom(A, 0.0, 100.0, 42);
    fillVectorRandom(B, 0.0, 100.0, 42);

    gemmCPU(C.data(), A.data(), B.data(), N, N, N);

    for (size_t i = 0; i < C.size(); i++) {
        std::cout << static_cast<float>(C[i]) << " ";
    }
    std::cout << "\n";

    return 0;
}