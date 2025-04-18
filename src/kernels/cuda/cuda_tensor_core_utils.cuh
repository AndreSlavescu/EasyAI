#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

/*
Synchronization and Barrier Device Functions
*/

__device__ __forceinline__ void threadblock_sync() {
    // threadblock-wide barrier
    asm volatile(
        "bar.sync 0;"
    );
}

__device__ __forceinline__ void warp_sync(uint32_t mask) {
    // all co-operating threads within the executing warp must reach this barrier.
    // For a full-mask, all threads are expected to reach the barrier.
    asm volatile(
        "bar.warp.sync %0;" 
        :: "r"(mask)
    );
}

/*
Shared Memory Layout Matrix Descriptor Device Functions
*/

__device__ __forceinline__ uint64_t matrix_descriptor_encode(uint64_t x) {
    return ((x & 0x3FFFF) >> 4);
}

__device__ __forceinline__ uint32_t compute_swizzle_base_offset(uint32_t pattern_start_addr) {
    // align shared memory layout depending on pattern type (32 byte, 64 byte, and 128 byte swizzle)
    return (pattern_start_addr >> 0x7) & 0x7;
}

template <typename DType>
__device__ uint64_t make_smem_desc_fp32(DType* generic_ptr) {
    uint32_t smem_addr = static_cast<uint32_t>(
        __cvta_generic_to_shared(generic_ptr)
    ); // converts generic address to shared memory address
    uint32_t aligned_smem_addr = compute_swizzle_base_offset(smem_addr);

    uint64_t matrix_descriptor = 0x0;
    matrix_descriptor |= matrix_descriptor_encode((uint64_t)aligned_smem_addr); // assign the addr to lower bits of descriptor
    matrix_descriptor |= matrix_descriptor_encode((uint64_t)0x10) << 16; // push a value of 16 at bit 16
    matrix_descriptor |= matrix_descriptor_encode((uint64_t)0x400) << 32; // push a value of 1024 at bit 32
    matrix_descriptor |= (uint64_t)0x1 << 62; // push a value of 1 at bit 62

    return matrix_descriptor;
}

/*
(Warp) Matrix Multiply Accumulate Device Functions
*/

__device__ __forceinline__ void wmma_m16n16k16_f16(
    float d_frag[8], // each thread's fragment from the 16x16 output matrix
    const half* a_frag,
    const half* b_frag
) {
    asm volatile(
        "{\n"
        "wmma.mma.m16n16k16.row.row.f32.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7},"
        " %8," // a_frag
        " %9;\n" // b_frag
        "}\n"
        : "+f"(d_frag[0]), "+f"(d_frag[1]), "+f"(d_frag[2]), "+f"(d_frag[3]),
          "+f"(d_frag[4]), "+f"(d_frag[5]), "+f"(d_frag[6]), "+f"(d_frag[7])
        : "r"(a_frag), "r"(b_frag)
    );
}

__device__ __forceinline__ void wmma_m16n16k16_bf16(
    float d_frag[8], // each thread's fragment from the 16x16 output matrix
    const bf16* a_frag,
    const bf16* b_frag
) {
    asm volatile(
        "{\n"
        "wmma.mma.m16n16k16.row.row.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7},"
        " %8," // a_frag
        " %9;\n" // b_frag
        "}\n"
        : "+f"(d_frag[0]), "+f"(d_frag[1]), "+f"(d_frag[2]), "+f"(d_frag[3]),
          "+f"(d_frag[4]), "+f"(d_frag[5]), "+f"(d_frag[6]), "+f"(d_frag[7])
        : "r"(a_frag), "r"(b_frag)
    );
}

template <typename DType>
__device__ __forceinline__ void wmma_m16n16k16_bf16(
    float d_frag[8], // each thread's fragment from the 16x16 output matrix
    const DType* a_frag,
    const DType* b_frag
) {
    if (std::is_same<DType, bf16>::value) {
        wmma_m16n16k16_bf16(
            d_frag,
            reinterpret_cast<const bf16*>(a_frag),
            reinterpret_cast<const bf16*>(b_frag)
        );
    } 
    else if (std::is_same<DType, half>::value) {
        wmma_m16n16k16_f16(
            d_frag,
            reinterpret_cast<const half*>(a_frag),
            reinterpret_cast<const half*>(b_frag)
        );
    } 
    else {
        assert(false && "Unsupported data type for wmma operation");
    }
}