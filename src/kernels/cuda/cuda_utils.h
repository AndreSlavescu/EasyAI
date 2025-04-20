#pragma once
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <cuda_runtime.h>
#include <random>

// kernel launch param helpers
#define MAX_THREADS_PER_BLOCK 1024
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// Warp size constant
#define WARP_SIZE 32

// CUDA_RESULT macro
#define CHECK_CUDA_RESULT(x) \
    do { CUresult result = x; if (result != CUDA_SUCCESS) { \
        const char* error_string; \
        cuGetErrorString(result, &error_string); \
        fprintf(stderr, "CUDA Driver Error: %s at %s line %d\n", error_string, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } } while(0)

// CUDA_ASSERT macro
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

// fill array of type T with random values
template <typename T>
void fillArrayRandom(T* vec, size_t size, T min, T max, unsigned int seed = std::random_device{}()) {
    std::mt19937 gen(seed);
    
    if constexpr (std::is_same_v<T, half>) {
        float min_f = __half2float(min);
        float max_f = __half2float(max);
        std::uniform_real_distribution<float> dist(min_f, max_f);
        for (size_t i = 0; i < size; ++i) {
            vec[i] = __float2half(dist(gen));
        }
    }
    else if constexpr (std::is_integral_v<T>) {
        std::uniform_int_distribution<T> dist(min, max);
        for (size_t i = 0; i < size; ++i) {
            vec[i] = dist(gen);
        }
    } else {
        std::uniform_real_distribution<T> dist(min, max);
        for (size_t i = 0; i < size; ++i) {
            vec[i] = dist(gen);
        }
    }
}

// fill vector of type T elements with random values
template <typename Vector>
void fillVectorRandom(Vector& vec, typename Vector::value_type min, 
                     typename Vector::value_type max, 
                     unsigned int seed = std::random_device{}()) {
    using T = typename Vector::value_type;
    fillArrayRandom(vec.data(), vec.size(), min, max, seed);
}