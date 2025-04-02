#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cuda_utils.h"

template <typename DType>
struct SmemTraits {
    static constexpr int alignment = sizeof(DType);
    static constexpr int modFactor = CEIL_DIV(sizeof(DType), 4);
};

template <typename DType>
__device__ __forceinline__ void smemWrite1DVal(
    DType* smem, 
    DType val,
    int max_idx = INT_MAX
) {
    constexpr int stride = SmemTraits<DType>::modFactor;

    if (threadIdx.x % stride == 0) {
        int idx = threadIdx.x / stride;
        if (idx < max_idx) {
            smem[idx] = val;
        }
        else {
            smem[idx] = DType(0);
        }
    }
}

template <typename DType, int Stride>
__device__ __forceinline__ void smemWrite2DVal(
    DType (*smem)[Stride],
    DType val,
    int max_rows = INT_MAX,
    int max_cols = INT_MAX
) {
    constexpr int spacing = SmemTraits<DType>::modFactor;

    if (threadIdx.y % spacing == 0) {
        int row = threadIdx.y / spacing;
        if (row < max_rows && threadIdx.x < max_cols) {
            smem[row][threadIdx.x] = val;
        }
        else {
            smem[row][threadIdx.x] = DType(0);
        }
    }
}

// barrier synchronization policy for tile load functions
struct WithSync { static constexpr bool sync = true; };
struct NoSync { static constexpr bool sync = false; };

template <typename DType, typename SyncPolicy = WithSync>
__device__ __forceinline__ void tileLoadFromGlobal1D(
    DType *smem,
    const DType* gmem,
    int origin_idx, // beginning index of tile
    int idxBound // total # of valid indices
) {
    unsigned int idx = origin_idx + threadIdx.x;

    DType val = gmem[idx];
    smemWrite1DVal<DType>(smem, val, idxBound);
    
    if constexpr (SyncPolicy::sync) {
        __syncthreads();
    }
}

template <typename DType, int Stride, typename SyncPolicy = WithSync>
__device__ __forceinline__ void tileLoadFromGlobal2D(
    DType (*smem)[Stride],
    const DType* gmem,
    int ld_gmem, // stride factor (row stride for rowmajor layout)
    int origin_row, // top-left row index this tile starts at
    int origin_col, // top-left col index this tile starts at
    int rowBound, // total # of valid rows
    int colBound // total # of valid cols
) {
    int row = origin_row + threadIdx.y;
    int col = origin_col + threadIdx.x;

    DType val = gmem[row * ld_gmem + col];
    smemWrite2DVal(smem, val, rowBound, colBound);

    if constexpr (SyncPolicy::sync) {
        __syncthreads();
    }
}
