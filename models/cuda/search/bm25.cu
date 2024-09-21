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
#include "../utils.h"

/*
    BM25 is a ranking function used by search engines to rank documents in their search results.

    Algorithm:
        Params:
            - Q is the query
            - D is the document
            - w_i is the weight of the i-th term in the query
            - R(q_i, d) is the relevance of the i-th term in the document
        score(Q, D) = \sum_{i=1}^{n} w_i \cdot R(q_i, d)
*/

#define DEBUG 0
#define PROFILING 0

void bm25_score_cpu(
    const float *query_weights,
    const float *document_weights,
    const float *query_lengths,
    const float *document_lengths,
    const float *average_document_length,
    const float k1,
    const float b,
    const int *query_indices,
    const int *document_indices,
    const int *query_terms,
    float *scores,
    const int num_queries,
    const int num_documents,
    const int num_terms
) {
    for (int i = 0; i < num_queries; i++) {
        float score = 0.0f;
        int query_start = query_indices[i];
        int query_end = query_indices[i + 1];

        for (int j = query_start; j < query_end; j++) {
            int term = query_terms[j];
            float qw_i = query_weights[term];
            float dw_i = document_weights[term];
            float inverse_document_freq = logf(1.0f + (num_documents - document_indices[term] + 0.5f) / (document_indices[term] + 0.5f));
            float term_frequency = dw_i / (k1 * (1.0f - b + b * (document_lengths[i] / average_document_length[0]))); 
            score += qw_i * inverse_document_freq * term_frequency;
        }
        scores[i] = score;
    }
}

__global__ void bm25_score_kernel(
    const float *query_weights,
    const float *document_weights,
    const float *query_lengths,
    const float *document_lengths,
    const float *average_document_length,
    const float k1,
    const float b,
    const int *query_indices,
    const int *document_indices,
    const int *query_terms,
    float *scores,
    const int num_queries,
    const int num_documents
) {
    extern __shared__ float shared_scores[];

    int query_idx = blockIdx.x;
    int term_idx = threadIdx.x;

    if (query_idx < num_queries) {
        float score = 0.0f;
        int query_start = query_indices[query_idx];
        int query_end = query_indices[query_idx + 1];
        int num_terms = query_end - query_start;

        if (term_idx < num_terms) {
            int term = query_terms[query_start + term_idx];
            float qw_i = query_weights[term]; // query_weights at term
            float dw_i = document_weights[term]; // document_weights at term
            float inverse_document_freq = logf(1.0f + (num_documents - document_indices[term] + 0.5f) / (document_indices[term] + 0.5f));
            float term_frequency = dw_i / (k1 * (1.0f - b + b * (document_lengths[query_idx] / average_document_length[0])));
            score = qw_i * inverse_document_freq * term_frequency;
        }

        shared_scores[term_idx] = score;
        __syncthreads();

        // Reduce partial scores to get the final score for the query
        if (term_idx == 0) {
            float final_score = 0.0f;
            for (int i = 0; i < num_terms; i++) {
                final_score += shared_scores[i];
            }
            scores[query_idx] = final_score;
        }
    }
}

void bm25_score(
    const float *query_weights,
    const float *document_weights,
    const float *query_lengths,
    const float *document_lengths,
    const float *average_document_length,
    const float k1,
    const float b,
    const int *query_indices,
    const int *document_indices,
    const int *query_terms,
    float *scores,
    const int num_queries,
    const int num_documents
) {
    int num_threads = MAX_THREADS_PER_BLOCK;
    int num_blocks = num_queries;

    // Create a cuda stream for efficient concurrent execution of the kernel
    cudaStream_t stream;
    CUDA_ASSERT(cudaStreamCreate(&stream) == cudaSuccess);

    // Launch the kernel with following parameters:
    //  - num_blocks: number of blocks (one block per query)
    //  - num_threads: number of threads per block (one thread per term)
    //  - smem_size: shared memory size
    //  - stream: stream identifier
    bm25_score_kernel<<<num_blocks, num_threads, num_threads * sizeof(float), stream>>>(
        query_weights,
        document_weights,
        query_lengths,
        document_lengths,
        average_document_length,
        k1,
        b,
        query_indices,
        document_indices,
        query_terms,
        scores,
        num_queries,
        num_documents
    );

    // Synchronize the stream to guarantee that all operations are completed correctly before returning
    CUDA_ASSERT(cudaStreamSynchronize(stream) == cudaSuccess);
    CUDA_ASSERT(cudaStreamDestroy(stream) == cudaSuccess);
}

int main() {
    const int num_queries = 1000; 
    const int num_documents = 1000;  
    const int num_terms = 3;

    // Vary query weights
    thrust::host_vector<float> h_query_weights(num_terms);
    for (int i = 0; i < num_terms; ++i) {
        h_query_weights[i] = 1.0f + i * 0.1f;  // Different weights for each term
    }

    // Vary document weights
    thrust::host_vector<float> h_document_weights(num_terms);
    for (int i = 0; i < num_terms; ++i) {
        h_document_weights[i] = 1.5f + i * 0.2f;  // Different weights for each term
    }

    // Vary query lengths
    thrust::host_vector<float> h_query_lengths(num_queries);
    for (int i = 0; i < num_queries; ++i) {
        h_query_lengths[i] = 3.0f + i * 0.1f;  // Different lengths for each query
    }

    // Vary document lengths
    thrust::host_vector<float> h_document_lengths(num_documents);
    for (int i = 0; i < num_documents; ++i) {
        h_document_lengths[i] = 10.0f + i * 0.2f;  // Different lengths for each document
    }

    thrust::host_vector<float> h_average_document_length = {11.0f};
    thrust::host_vector<float> h_k1 = {1.2f};
    thrust::host_vector<float> h_b = {0.75f};

    // Define query_indices and document_indices for Compressed Sparse Row (CSR) structure.
    thrust::host_vector<int> h_query_indices(num_queries + 1);
    thrust::host_vector<int> h_document_indices(num_documents + 1);
    for (int i = 0; i <= num_queries; ++i) {
        h_query_indices[i] = i * num_terms;
    }
    for (int i = 0; i <= num_documents; ++i) {
        h_document_indices[i] = i * num_terms;
    }

    // Define term indices for each query for each document.
    thrust::host_vector<int> h_query_terms(num_queries * num_terms);
    thrust::host_vector<int> h_document_terms(num_documents * num_terms);
    for (int i = 0; i < num_queries * num_terms; ++i) {
        h_query_terms[i] = i % num_terms;
    }
    for (int i = 0; i < num_documents * num_terms; ++i) {
        h_document_terms[i] = i % num_terms;
    }

    // Query weights
    thrust::device_vector<float> d_query_weights = h_query_weights;

    // Document weights
    thrust::device_vector<float> d_document_weights = h_document_weights;

    // Query lengths
    thrust::device_vector<float> d_query_lengths = h_query_lengths;

    // Document lengths
    thrust::device_vector<float> d_document_lengths = h_document_lengths;

    // Average document length
    thrust::device_vector<float> d_average_document_length = h_average_document_length;

    // ranking function sensitivity parameter    
    thrust::device_vector<float> d_k1 = h_k1;

    // length normalization parameter
    thrust::device_vector<float> d_b = h_b;

    // Query indices
    thrust::device_vector<int> d_query_indices = h_query_indices;

    // Document indices
    thrust::device_vector<int> d_document_indices = h_document_indices;

    // Query terms
    thrust::device_vector<int> d_query_terms = h_query_terms;

    // Document terms
    thrust::device_vector<int> d_document_terms = h_document_terms;

    // Scores
    thrust::device_vector<float> d_scores(num_queries, 0.0f);
    thrust::host_vector<float> h_scores(num_queries, 0.0f);

    bm25_score(
        thrust::raw_pointer_cast(d_query_weights.data()),
        thrust::raw_pointer_cast(d_document_weights.data()),
        thrust::raw_pointer_cast(d_query_lengths.data()),
        thrust::raw_pointer_cast(d_document_lengths.data()),
        thrust::raw_pointer_cast(d_average_document_length.data()),
        h_k1[0],
        h_b[0],
        thrust::raw_pointer_cast(d_query_indices.data()),
        thrust::raw_pointer_cast(d_document_indices.data()),
        thrust::raw_pointer_cast(d_query_terms.data()),
        thrust::raw_pointer_cast(d_scores.data()),
        num_queries,
        num_documents
    );

    // Device to host copy
    CUDA_ASSERT(cudaMemcpy(h_scores.data(), thrust::raw_pointer_cast(d_scores.data()), num_queries * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);

    // Compute expected scores using CPU implementation
    thrust::host_vector<float> expected_scores(num_queries, 0.0f);
    bm25_score_cpu(
        h_query_weights.data(),
        h_document_weights.data(),
        h_query_lengths.data(),
        h_document_lengths.data(),
        h_average_document_length.data(),
        h_k1[0],
        h_b[0],
        h_query_indices.data(),
        h_document_indices.data(),
        h_query_terms.data(),
        expected_scores.data(),
        num_queries,
        num_documents,
        num_terms
    );

    // Compare GPU results with CPU results
    for (int i = 0; i < num_queries; ++i) {
        if (DEBUG) {
            std::cout << "Query " << i << " Score: " << h_scores[i] << std::endl;
            std::cout << "Expected Score: " << expected_scores[i] << std::endl;
            std::cout << "Difference: " << abs(h_scores[i] - expected_scores[i]) << std::endl;
        }
        assert(abs(h_scores[i] - expected_scores[i]) < 1e-5);
    }

    if (PROFILING) {
        // CPU implementation profiling
        thrust::host_vector<float> h_cpu_scores(num_queries, 0.0f);
        auto start_cpu = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; ++i) {
            bm25_score_cpu(
                h_query_weights.data(),
                h_document_weights.data(),
                h_query_lengths.data(),
                h_document_lengths.data(),
                h_average_document_length.data(),
                h_k1[0],
                h_b[0],
                h_query_indices.data(),
                h_document_indices.data(),
                h_query_terms.data(),
                h_cpu_scores.data(),
                num_queries,
                num_documents,
                num_terms
            );
        }
        auto end_cpu = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> cpu_duration = end_cpu - start_cpu;
        std::cout << "CPU BM25 Score computation time (100 iterations): " << cpu_duration.count() / 100 << " seconds per iteration" << std::endl;

        // GPU implementation profiling
        auto start_gpu = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; ++i) {
            bm25_score(
                thrust::raw_pointer_cast(d_query_weights.data()),
                thrust::raw_pointer_cast(d_document_weights.data()),
                thrust::raw_pointer_cast(d_query_lengths.data()),
                thrust::raw_pointer_cast(d_document_lengths.data()),
                thrust::raw_pointer_cast(d_average_document_length.data()),
                h_k1[0],
                h_b[0],
                thrust::raw_pointer_cast(d_query_indices.data()),
                thrust::raw_pointer_cast(d_document_indices.data()),
                thrust::raw_pointer_cast(d_query_terms.data()),
                thrust::raw_pointer_cast(d_scores.data()),
                num_queries,
                num_documents
            );
            CUDA_ASSERT(cudaStreamSynchronize(0) == cudaSuccess);
        }
        auto end_gpu = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> gpu_duration = end_gpu - start_gpu;
        std::cout << "GPU BM25 Score computation time (100 iterations): " << gpu_duration.count() / 100 << " seconds per iteration" << std::endl;
    }

    return 0;
}