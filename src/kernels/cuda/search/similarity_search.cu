#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/scan.h>
#include <thrust/extrema.h>
#include <iostream>
#include <random>
#include <chrono>
#include "../cuda_utils.h"

#define DEBUG 0
#define PROFILING 0

struct SimilarityResult {
    int document_index;
    float similarity_score;
};

SimilarityResult similarity_search_cpu(
    const float *query_embeddings,
    const float *document_embeddings,
    const int num_queries,
    const int num_documents,
    const int embedding_dim,
    const float score_threshold = 0.75f
) {
    int most_similar_document_index = -1;
    float highest_similarity_score = -1.0f;

    for (int i = 0; i < num_queries; i++) {
        for (int j = 0; j < num_documents; j++) {
            float dot_product = 0.0f;
            float vec1_dp = 0.0f;
            float vec2_dp = 0.0f;

            for (int k = 0; k < embedding_dim; k++) {
                float q = query_embeddings[i * embedding_dim + k];
                float d = document_embeddings[j * embedding_dim + k];
                dot_product += q * d;
                vec1_dp += q * q;
                vec2_dp += d * d;
            }

            float similarity = (vec1_dp > 0.0f && vec2_dp > 0.0f) ?
                               (dot_product / (sqrtf(vec1_dp) * sqrtf(vec2_dp))) : 0.0f;

            if (similarity > highest_similarity_score && similarity >= score_threshold) {
                highest_similarity_score = similarity;
                most_similar_document_index = j;
            }
        }
    }

    return SimilarityResult{most_similar_document_index, highest_similarity_score};
}

__global__ void similarity_search_kernel(
    const float *query_embeddings,
    const float *document_embeddings,
    float *similarities,
    int *indices,
    const int num_queries,
    const int num_documents,
    const int embedding_dim,
    const float score_threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = num_queries * num_documents;

    if (idx < total_pairs) {
        int query_idx = idx / num_documents;
        int doc_idx = idx % num_documents;

        float dot_product = 0.0f;
        float vec1_dp = 0.0f;
        float vec2_dp = 0.0f;

        for (int i = 0; i < embedding_dim; ++i) {
            float q = query_embeddings[query_idx * embedding_dim + i];
            float d = document_embeddings[doc_idx * embedding_dim + i];
            dot_product += q * d;
            vec1_dp += q * q;
            vec2_dp += d * d;
        }

        float similarity = 0.0f;
        if (vec1_dp > 0.0f && vec2_dp > 0.0f) {
            similarity = dot_product / (sqrtf(vec1_dp) * sqrtf(vec2_dp));
        }

        if (similarity >= score_threshold) {
            similarities[idx] = similarity;
            indices[idx] = doc_idx;
        } else {
            similarities[idx] = -1.0f;
            indices[idx] = -1;
        }
    }
}

void similarity_search(
    const float *query_embeddings,
    const float *document_embeddings,
    const int num_queries,
    const int num_documents,
    const int embedding_dim,
    const float score_threshold,
    SimilarityResult *result
) {
    int total_pairs = num_queries * num_documents;

    thrust::device_vector<float> d_similarities(total_pairs);
    thrust::device_vector<int> d_indices(total_pairs);

    int threads_per_block = 256;
    int blocks_per_grid = (total_pairs + threads_per_block - 1) / threads_per_block;

    similarity_search_kernel<<<blocks_per_grid, threads_per_block>>>(
        query_embeddings,
        document_embeddings,
        thrust::raw_pointer_cast(d_similarities.data()),
        thrust::raw_pointer_cast(d_indices.data()),
        num_queries,
        num_documents,
        embedding_dim,
        score_threshold
    );

    CUDA_ASSERT(cudaGetLastError());
    CUDA_ASSERT(cudaDeviceSynchronize());

    auto max_elem = thrust::max_element(d_similarities.begin(), d_similarities.end());
    float max_similarity = *max_elem;
    int max_index = max_elem - d_similarities.begin();

    int document_index = -1;
    if (max_similarity >= score_threshold) {
        document_index = d_indices[max_index];
    }

    result->similarity_score = max_similarity;
    result->document_index = document_index;
}

int main() {
    const int num_queries = 10;
    const int num_documents = 10;
    const int embedding_dim = 768;
    const float score_threshold = 0.75f;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    thrust::host_vector<float> h_query_embeddings(num_queries * embedding_dim);
    thrust::host_vector<float> h_document_embeddings(num_documents * embedding_dim);

    auto randomize_embeddings = [&]() {
        for (auto &val : h_query_embeddings) {
            val = dis(gen);
        }
        for (auto &val : h_document_embeddings) {
            val = dis(gen);
        }
    };

    thrust::device_vector<float> d_query_embeddings(num_queries * embedding_dim);
    thrust::device_vector<float> d_document_embeddings(num_documents * embedding_dim);

    const int num_tests = 5;

    for (int test = 0; test < num_tests; ++test) {
        randomize_embeddings();
        thrust::copy(h_query_embeddings.begin(), h_query_embeddings.end(), d_query_embeddings.begin());
        thrust::copy(h_document_embeddings.begin(), h_document_embeddings.end(), d_document_embeddings.begin());

        SimilarityResult cpu_result = similarity_search_cpu(
            h_query_embeddings.data(),
            h_document_embeddings.data(),
            num_queries,
            num_documents,
            embedding_dim,
            score_threshold
        );

        SimilarityResult gpu_result;
        similarity_search(
            thrust::raw_pointer_cast(d_query_embeddings.data()),
            thrust::raw_pointer_cast(d_document_embeddings.data()),
            num_queries,
            num_documents,
            embedding_dim,
            score_threshold,
            &gpu_result
        );

        if (DEBUG) {
            if (gpu_result.document_index != cpu_result.document_index ||
                fabs(gpu_result.similarity_score - cpu_result.similarity_score) > 1e-5) {
                std::cout << "Mismatch: "
                          << "CPU index = " << cpu_result.document_index
                          << ", GPU index = " << gpu_result.document_index
                          << "; CPU score = " << cpu_result.similarity_score
                          << ", GPU score = " << gpu_result.similarity_score << std::endl;
            } else {
                std::cout << "GPU result matches CPU result." << std::endl;
                std::cout << "Document index: " << gpu_result.document_index
                          << ", Similarity score: " << gpu_result.similarity_score << std::endl;
            }
        }
    }

    if (PROFILING) {
        auto start_cpu = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; ++i) {
            randomize_embeddings();
            similarity_search_cpu(
                h_query_embeddings.data(),
                h_document_embeddings.data(),
                num_queries,
                num_documents,
                embedding_dim,
                score_threshold
            );
        }
        auto end_cpu = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> cpu_duration = end_cpu - start_cpu;
        std::cout << "CPU similarity search time (100 iterations): "
                  << cpu_duration.count() / 100 << " seconds per iteration" << std::endl;

        auto start_gpu = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; ++i) {
            randomize_embeddings();
            thrust::copy(h_query_embeddings.begin(), h_query_embeddings.end(), d_query_embeddings.begin());
            thrust::copy(h_document_embeddings.begin(), h_document_embeddings.end(), d_document_embeddings.begin());

            SimilarityResult gpu_result;
            similarity_search(
                thrust::raw_pointer_cast(d_query_embeddings.data()),
                thrust::raw_pointer_cast(d_document_embeddings.data()),
                num_queries,
                num_documents,
                embedding_dim,
                score_threshold,
                &gpu_result
            );
        }
        auto end_gpu = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> gpu_duration = end_gpu - start_gpu;
        std::cout << "GPU similarity search time (100 iterations): "
                  << gpu_duration.count() / 100 << " seconds per iteration" << std::endl;
    }

    return 0;
}