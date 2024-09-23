#pragma once
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <cuda_runtime.h>

// kernel launch param helpers
#define MAX_THREADS_PER_BLOCK 1024
#define CEIL(a, b) (((a) + (b) - 1) / (b))
#define DIV_CEIL(a, b) (((a) % (b) == 0) ? (a) / (b) : (a) / (b) + 1)

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
