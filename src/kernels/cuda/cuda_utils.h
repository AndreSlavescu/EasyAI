#pragma once
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <cuda_runtime.h>

// kernel launch param helpers
#define MAX_THREADS_PER_BLOCK 1024
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// Basic CUDA_ASSERT macro
#define CUDA_ASSERT(condition) \
    do { \
        cudaError_t error = condition; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
                    __FILE__, __LINE__, static_cast<unsigned int>(error), \
                    cudaGetErrorString(error), #condition); \
            cudaDeviceReset(); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CUBLAS_ASSERT(condition) \
    do { \
        cublasStatus_t status = condition; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d code=%d\n", \
                    __FILE__, __LINE__, static_cast<int>(status)); \
            cudaDeviceReset(); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)
