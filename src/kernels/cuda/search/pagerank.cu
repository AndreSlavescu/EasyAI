#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/scan.h>
#include <iostream>
#include <random>
#include <chrono>
#include "../cuda_utils.h"

/*
    PageRank is an algorithm used by search engines to rank web pages in their search results.

    Algorithm:
        Params:
            - PR(p_i) is the pagerank of page i
            - N is the total number of pages
            - M(p_i) is the set of pages linking to page i
            - L(p_j) is the number of outgoing page links from page j
            - d is damping factor
                - damping factor denotes the probability that a user will continue down a chain of links
        PR(p_i) = (1 - d) / N + d * \sum_{p_j \in M(p_i)} PR(p_j) / L(p_j)
*/

#define DEBUG 0
#define PROFILING 0

void pagerank_cpu(
    float* ranked_pages,
    float* pages,
    int num_pages,
    float damping_factor
) {
    const float norm = (1 - damping_factor) / num_pages;
    for (int i = 0; i < num_pages; i++) {
        float sum = 0.0f;
        for (int j = 0; j < num_pages; j++) {
            if (pages[j * num_pages + i] > 0) {
                sum += ranked_pages[j] / pages[j * num_pages + i];
            }
        }
        ranked_pages[i] = norm + damping_factor * sum;
    }
}

__global__ void pagerank_kernel(
    float* ranked_pages,
    const float* __restrict__ pages,
    int num_pages,
    float damping_factor
) {
    const float norm = (1 - damping_factor) / num_pages;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_pages) {
        float sum = 0.0f;
        for (int j = 0; j < num_pages; j++) {
            if (pages[j * num_pages + idx] > 0) {
                sum += ranked_pages[j] / pages[j * num_pages + idx];
            }
        }
        ranked_pages[idx] = norm + damping_factor * sum;
    }
}

int main() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 9);

    const int num_pages = 1000;
    const float damping_factor = 0.85f;
    thrust::host_vector<float> h_ranked_pages(num_pages, 1.0f);
    thrust::host_vector<float> h_pages(num_pages * num_pages);

    for (int i = 0; i < num_pages; i++) {
        for (int j = 0; j < num_pages; j++) {
            h_pages[i * num_pages + j] = (i != j && dis(gen) < 3) ? 1 : 0;
        }
    }

    if (PROFILING) {
        auto start_cpu = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; ++i) {
            pagerank_cpu(h_ranked_pages.data(), h_pages.data(), num_pages, damping_factor);
        }
        auto end_cpu = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> cpu_duration = end_cpu - start_cpu;
        std::cout << "CPU PageRank computation time (100 iterations): " << cpu_duration.count() / 100 << " seconds per iteration" << std::endl;
    }

    thrust::device_vector<float> d_ranked_pages = h_ranked_pages;
    thrust::device_vector<float> d_pages = h_pages;

    int threads_per_block = 256;
    int blocks_per_grid = (num_pages + threads_per_block - 1) / threads_per_block;

    if (PROFILING) {
        auto start_gpu = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; ++i) {
            pagerank_kernel<<<blocks_per_grid, threads_per_block>>>(
                thrust::raw_pointer_cast(d_ranked_pages.data()),
                thrust::raw_pointer_cast(d_pages.data()),
                num_pages,
                damping_factor
            );
            CUDA_ASSERT(cudaGetLastError());
            CUDA_ASSERT(cudaDeviceSynchronize());
        }
        auto end_gpu = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> gpu_duration = end_gpu - start_gpu;
        std::cout << "GPU PageRank computation time (100 iterations): " << gpu_duration.count() / 100 << " seconds per iteration" << std::endl;
    } else {
        pagerank_kernel<<<blocks_per_grid, threads_per_block>>>(
            thrust::raw_pointer_cast(d_ranked_pages.data()),
            thrust::raw_pointer_cast(d_pages.data()),
            num_pages,
            damping_factor
        );
        CUDA_ASSERT(cudaGetLastError());
        CUDA_ASSERT(cudaDeviceSynchronize());
    }

    thrust::copy(d_ranked_pages.begin(), d_ranked_pages.end(), h_ranked_pages.begin());
    for (int i = 0; i < num_pages; i++) {
        if (DEBUG) {
            std::cout << "PageRank of page " << i << ": " << h_ranked_pages[i] << std::endl;
        }
    }

    return 0;
}