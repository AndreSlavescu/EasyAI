#include <immintrin.h>
#include <iostream>
#include <vector>
#include <random>
#include <climits>
#include "../cpu_avx_utils.h"

#define DEBUG 0
#define PROFILING 0

void matmul_nxnxn_baseline(
    float* a, 
    float* b, 
    float* c,
    int n
) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            c[i * n + j] = 0.0f;
            for (int k = 0; k < n; k++) {
                c[i * n + j] += a[i * n + k] * b[n * k + j];
            }
        }
    }
}

void matmul_nxnxn_avx_128bit(
    float* a,
    float* b,
    float* c, 
    int n
) {
    for (int i = 0; i < n; i += 4) {
        for (int j = 0; j < n; j += 4) {
            __m128 c_row1 = _mm_setzero_ps();
            __m128 c_row2 = _mm_setzero_ps();
            __m128 c_row3 = _mm_setzero_ps();
            __m128 c_row4 = _mm_setzero_ps();
            for (int k = 0; k < n; ++k) {
                __m128 a_elem1 = _mm_set1_ps(a[(i + 0) * n + k]);
                __m128 a_elem2 = _mm_set1_ps(a[(i + 1) * n + k]);
                __m128 a_elem3 = _mm_set1_ps(a[(i + 2) * n + k]);
                __m128 a_elem4 = _mm_set1_ps(a[(i + 3) * n + k]);
                __m128 b_row = _mm_load_ps(&b[k * n + j]); 
                c_row1 = _mm_add_ps(c_row1, _mm_mul_ps(a_elem1, b_row));
                c_row2 = _mm_add_ps(c_row2, _mm_mul_ps(a_elem2, b_row));
                c_row3 = _mm_add_ps(c_row3, _mm_mul_ps(a_elem3, b_row));
                c_row4 = _mm_add_ps(c_row4, _mm_mul_ps(a_elem4, b_row));
            }
            _mm_store_ps(&c[(i + 0) * n + j], c_row1);
            _mm_store_ps(&c[(i + 1) * n + j], c_row2);
            _mm_store_ps(&c[(i + 2) * n + j], c_row3);
            _mm_store_ps(&c[(i + 3) * n + j], c_row4);
        }
    }
}

int main() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    std::vector<int> test_sizes(5);
    for (size_t i = 0; i < test_sizes.size(); i++) {
        test_sizes[i] = pow(2, (5 + i));  
    }

    for (const int n : test_sizes) {
        std::vector<float> matrix_a(n * n);
        std::vector<float> matrix_b(n * n);
        std::vector<float> dst_standard(n * n);
        std::vector<float> dst_avx_128bit(n * n);

        for (float &val : matrix_a) {
            val = dis(gen);
        }

        for (float &val : matrix_b) {
            val = dis(gen);
        }

        if (PROFILING) {
            utils::compare_performance(
                "Standard sgemm", 
                [&](){ matmul_nxnxn_baseline(matrix_a.data(), matrix_b.data(), dst_standard.data(), n); },
                "AVX 128-bit sgemm", 
                [&](){ matmul_nxnxn_avx_128bit(matrix_a.data(), matrix_b.data(), dst_avx_128bit.data(), n); },
                100
            );
        }

        bool is_correct_128bit = utils::check_correctness(dst_standard, dst_avx_128bit);

        if (DEBUG) {
            std::cout << "Correctness check (128-bit AVX): " << (is_correct_128bit ? "PASSED" : "FAILED") << std::endl;
        }

        if (!is_correct_128bit) return EXIT_FAILURE;
    }

    return 0;
}
