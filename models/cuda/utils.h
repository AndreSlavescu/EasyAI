#pragma once
#include <cstdio>
#include <cassert>

// kernel launch param helpers
#define MAX_THREADS_PER_BLOCK 1024
#define CEIL(a, b) (((a) + (b) - 1) / (b))
#define DIV_CEIL(a, b) (((a) % (b) == 0) ? (a) / (b) : (a) / (b) + 1)

// Basic assertion for CUDA kernels
#define CUDA_ASSERT(condition)                                              \
    do {                                                                    \
        if (!(condition)) {                                                 \
            fprintf(stderr, "CUDA_ASSERT failed: %s, file %s, line %d\n",   \
                    #condition, __FILE__, __LINE__);                        \
            assert(condition);                                              \
        }                                                                   \
    } while (0)
