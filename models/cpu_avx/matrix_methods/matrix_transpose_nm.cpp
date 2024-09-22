#include <immintrin.h>
#include <iostream>
#include <vector>
#include <random>
#include <climits>
#include "../utils.h"

#define DEBUG 0
#define PROFILING 0

void transpose_nxm_baseline(
    float* src, 
    float* dst, 
    int n,
    int m
) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            dst[j * n + i] = src[i * m + j];
        }
    }
}

void transpose_nxm_avx2_128bit(
    float* src, 
    float* dst, 
    int n,
    int m
) {
    for (int i = 0; i < n; i += 4) {
        for (int j = 0; j < m; j += 4) {
            __m128 row1 = _mm_load_ps(&src[i * m + j]);
            __m128 row2 = _mm_load_ps(&src[(i + 1) * m + j]);
            __m128 row3 = _mm_load_ps(&src[(i + 2) * m + j]);
            __m128 row4 = _mm_load_ps(&src[(i + 3) * m + j]);

            _MM_TRANSPOSE4_PS(row1, row2, row3, row4);

            _mm_store_ps(&dst[j * n + i], row1);
            _mm_store_ps(&dst[(j + 1) * n + i], row2);
            _mm_store_ps(&dst[(j + 2) * n + i], row3);
            _mm_store_ps(&dst[(j + 3) * n + i], row4);
        }
    }
}

void transpose_nxm_avx2_256bit_kernel1(
    const float* __restrict__ src, 
    float* dst, 
    int n,
    int m
) {
    for (int i = 0; i < n; i += 8) {
        for (int j = 0; j < m; j += 8) {
            __m256 row0 = _mm256_loadu_ps(&src[i * m + j]);
            __m256 row1 = _mm256_loadu_ps(&src[(i + 1) * m + j]);
            __m256 row2 = _mm256_loadu_ps(&src[(i + 2) * m + j]);
            __m256 row3 = _mm256_loadu_ps(&src[(i + 3) * m + j]);
            __m256 row4 = _mm256_loadu_ps(&src[(i + 4) * m + j]);
            __m256 row5 = _mm256_loadu_ps(&src[(i + 5) * m + j]);
            __m256 row6 = _mm256_loadu_ps(&src[(i + 6) * m + j]);
            __m256 row7 = _mm256_loadu_ps(&src[(i + 7) * m + j]);

            __m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;
            __m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;
            __t0 = _mm256_unpacklo_ps(row0, row1);
            __t1 = _mm256_unpackhi_ps(row0, row1);
            __t2 = _mm256_unpacklo_ps(row2, row3);
            __t3 = _mm256_unpackhi_ps(row2, row3);
            __t4 = _mm256_unpacklo_ps(row4, row5);
            __t5 = _mm256_unpackhi_ps(row4, row5);
            __t6 = _mm256_unpacklo_ps(row6, row7);
            __t7 = _mm256_unpackhi_ps(row6, row7);
            __tt0 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(1,0,1,0));
            __tt1 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(3,2,3,2));
            __tt2 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(1,0,1,0));
            __tt3 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(3,2,3,2));
            __tt4 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(1,0,1,0));
            __tt5 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(3,2,3,2));
            __tt6 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(1,0,1,0));
            __tt7 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(3,2,3,2));
            row0 = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);
            row1 = _mm256_permute2f128_ps(__tt1, __tt5, 0x20);
            row2 = _mm256_permute2f128_ps(__tt2, __tt6, 0x20);
            row3 = _mm256_permute2f128_ps(__tt3, __tt7, 0x20);
            row4 = _mm256_permute2f128_ps(__tt0, __tt4, 0x31);
            row5 = _mm256_permute2f128_ps(__tt1, __tt5, 0x31);
            row6 = _mm256_permute2f128_ps(__tt2, __tt6, 0x31);
            row7 = _mm256_permute2f128_ps(__tt3, __tt7, 0x31);

            _mm256_storeu_ps(&dst[j * n + i], row0);
            _mm256_storeu_ps(&dst[(j + 1) * n + i], row1);
            _mm256_storeu_ps(&dst[(j + 2) * n + i], row2);
            _mm256_storeu_ps(&dst[(j + 3) * n + i], row3);
            _mm256_storeu_ps(&dst[(j + 4) * n + i], row4);
            _mm256_storeu_ps(&dst[(j + 5) * n + i], row5);
            _mm256_storeu_ps(&dst[(j + 6) * n + i], row6);
            _mm256_storeu_ps(&dst[(j + 7) * n + i], row7);
        }
    }
}

/*
kernel2 improves on kernel1 by doing simple blocktiling, where each tile is 16x16.
There is an average speedup increase of 0.3x over kernel1.

Tested on i7-6800k @ 3.40GHz
    kernel1 speedup against baseline: ~5.9x
    kernel2 speedup against baseline: ~6.2x
*/
void transpose_nxm_avx2_256bit_kernel2(
    const float* __restrict__ src, 
    float* dst, 
    int n,
    int m
) {
    const int tile_size = 16;

    for (int i = 0; i < n; i += tile_size) {
        for (int j = 0; j < m; j += tile_size) {
            for (int ti = 0; ti < tile_size; ti += 8) {
                for (int tj = 0; tj < tile_size; tj += 8) {
                    if (i + ti + 7 < n && j + tj + 7 < m) {
                        __m256 row0 = _mm256_loadu_ps(&src[(i + ti) * m + (j + tj)]);
                        __m256 row1 = _mm256_loadu_ps(&src[(i + ti + 1) * m + (j + tj)]);
                        __m256 row2 = _mm256_loadu_ps(&src[(i + ti + 2) * m + (j + tj)]);
                        __m256 row3 = _mm256_loadu_ps(&src[(i + ti + 3) * m + (j + tj)]);
                        __m256 row4 = _mm256_loadu_ps(&src[(i + ti + 4) * m + (j + tj)]);
                        __m256 row5 = _mm256_loadu_ps(&src[(i + ti + 5) * m + (j + tj)]);
                        __m256 row6 = _mm256_loadu_ps(&src[(i + ti + 6) * m + (j + tj)]);
                        __m256 row7 = _mm256_loadu_ps(&src[(i + ti + 7) * m + (j + tj)]);

                        __m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;
                        __m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;
                        __t0 = _mm256_unpacklo_ps(row0, row1);
                        __t1 = _mm256_unpackhi_ps(row0, row1);
                        __t2 = _mm256_unpacklo_ps(row2, row3);
                        __t3 = _mm256_unpackhi_ps(row2, row3);
                        __t4 = _mm256_unpacklo_ps(row4, row5);
                        __t5 = _mm256_unpackhi_ps(row4, row5);
                        __t6 = _mm256_unpacklo_ps(row6, row7);
                        __t7 = _mm256_unpackhi_ps(row6, row7);
                        __tt0 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(1,0,1,0));
                        __tt1 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(3,2,3,2));
                        __tt2 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(1,0,1,0));
                        __tt3 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(3,2,3,2));
                        __tt4 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(1,0,1,0));
                        __tt5 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(3,2,3,2));
                        __tt6 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(1,0,1,0));
                        __tt7 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(3,2,3,2));
                        row0 = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);
                        row1 = _mm256_permute2f128_ps(__tt1, __tt5, 0x20);
                        row2 = _mm256_permute2f128_ps(__tt2, __tt6, 0x20);
                        row3 = _mm256_permute2f128_ps(__tt3, __tt7, 0x20);
                        row4 = _mm256_permute2f128_ps(__tt0, __tt4, 0x31);
                        row5 = _mm256_permute2f128_ps(__tt1, __tt5, 0x31);
                        row6 = _mm256_permute2f128_ps(__tt2, __tt6, 0x31);
                        row7 = _mm256_permute2f128_ps(__tt3, __tt7, 0x31);

                        _mm256_storeu_ps(&dst[(j + tj) * n + (i + ti)], row0);
                        _mm256_storeu_ps(&dst[(j + tj + 1) * n + (i + ti)], row1);
                        _mm256_storeu_ps(&dst[(j + tj + 2) * n + (i + ti)], row2);
                        _mm256_storeu_ps(&dst[(j + tj + 3) * n + (i + ti)], row3);
                        _mm256_storeu_ps(&dst[(j + tj + 4) * n + (i + ti)], row4);
                        _mm256_storeu_ps(&dst[(j + tj + 5) * n + (i + ti)], row5);
                        _mm256_storeu_ps(&dst[(j + tj + 6) * n + (i + ti)], row6);
                        _mm256_storeu_ps(&dst[(j + tj + 7) * n + (i + ti)], row7);
                    }
                }
            }
        }
    }
}

/*
kernel3 improves on kernel2 by using Morton ordering for texture swizzling.
Morton ordering helps in improving cache locality by accessing memory in a pattern that is more makes use of cache-lines more effectively.

Let's use a simple example of a 4x4 tile, which in our case happens to be the tile size anyway:
 
Original Order:
  0  1  2  3
  4  5  6  7
  8  9 10 11
 
Morton Order:
  0  1  4  5
  2  3  6  7
  8  9 12 13

In the original order, accessing elements sequentially in row-major, which is not optimal for cache access.
By processing tiles in column-major, cache lines can be more effecitvely used, which results in fewer cache misses.

Cache access pattern:
0 -> 1 -> 4 -> 5 -> 2 -> 3 -> 6 -> 7 -> 8 -> 9 -> 12 -> 13

Tested on i7-6800k @ 3.40GHz
    kernel1 speedup against baseline: ~5.9x
    kernel2 speedup against baseline: ~6.2x
    kernel3 speedup against baseline: ~6.7x
*/
void transpose_nxm_avx2_256bit_kernel3(
    const float* __restrict__ src,
    float* __restrict__ dst, 
    int n,
    int m
) {
    const int tile_size = 16;
    int num_tiles_n = (n + tile_size - 1) / tile_size;
    int num_tiles_m = (m + tile_size - 1) / tile_size;
    int total_tiles = num_tiles_n * num_tiles_m;

    for (int tile_index = 0; tile_index < total_tiles; ++tile_index) {
        unsigned int x = 0, y = 0;
        unsigned int t = tile_index;
        for (unsigned int i = 0; i < sizeof(unsigned int) * CHAR_BIT / 2; ++i) {
            x |= ((t & (1 << (2 * i))) >> i);
            y |= ((t & (1 << (2 * i + 1))) >> (i + 1));
        }
        int i = x * tile_size;
        int j = y * tile_size;

        if (i < n && j < m) {
            for (int ti = 0; ti < tile_size; ti += 8) {
                for (int tj = 0; tj < tile_size; tj += 8) {
                    if (i + ti + 7 < n && j + tj + 7 < m) {
                        __m256 row0 = _mm256_loadu_ps(&src[(i + ti) * m + (j + tj)]);
                        __m256 row1 = _mm256_loadu_ps(&src[(i + ti + 1) * m + (j + tj)]);
                        __m256 row2 = _mm256_loadu_ps(&src[(i + ti + 2) * m + (j + tj)]);
                        __m256 row3 = _mm256_loadu_ps(&src[(i + ti + 3) * m + (j + tj)]);
                        __m256 row4 = _mm256_loadu_ps(&src[(i + ti + 4) * m + (j + tj)]);
                        __m256 row5 = _mm256_loadu_ps(&src[(i + ti + 5) * m + (j + tj)]);
                        __m256 row6 = _mm256_loadu_ps(&src[(i + ti + 6) * m + (j + tj)]);
                        __m256 row7 = _mm256_loadu_ps(&src[(i + ti + 7) * m + (j + tj)]);

                        __m256 __t0 = _mm256_unpacklo_ps(row0, row1);
                        __m256 __t1 = _mm256_unpackhi_ps(row0, row1);
                        __m256 __t2 = _mm256_unpacklo_ps(row2, row3);
                        __m256 __t3 = _mm256_unpackhi_ps(row2, row3);
                        __m256 __t4 = _mm256_unpacklo_ps(row4, row5);
                        __m256 __t5 = _mm256_unpackhi_ps(row4, row5);
                        __m256 __t6 = _mm256_unpacklo_ps(row6, row7);
                        __m256 __t7 = _mm256_unpackhi_ps(row6, row7);

                        __m256 __tt0 = _mm256_shuffle_ps(__t0, __t2, _MM_SHUFFLE(1, 0, 1, 0));
                        __m256 __tt1 = _mm256_shuffle_ps(__t0, __t2, _MM_SHUFFLE(3, 2, 3, 2));
                        __m256 __tt2 = _mm256_shuffle_ps(__t1, __t3, _MM_SHUFFLE(1, 0, 1, 0));
                        __m256 __tt3 = _mm256_shuffle_ps(__t1, __t3, _MM_SHUFFLE(3, 2, 3, 2));
                        __m256 __tt4 = _mm256_shuffle_ps(__t4, __t6, _MM_SHUFFLE(1, 0, 1, 0));
                        __m256 __tt5 = _mm256_shuffle_ps(__t4, __t6, _MM_SHUFFLE(3, 2, 3, 2));
                        __m256 __tt6 = _mm256_shuffle_ps(__t5, __t7, _MM_SHUFFLE(1, 0, 1, 0));
                        __m256 __tt7 = _mm256_shuffle_ps(__t5, __t7, _MM_SHUFFLE(3, 2, 3, 2));

                        row0 = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);
                        row1 = _mm256_permute2f128_ps(__tt1, __tt5, 0x20);
                        row2 = _mm256_permute2f128_ps(__tt2, __tt6, 0x20);
                        row3 = _mm256_permute2f128_ps(__tt3, __tt7, 0x20);
                        row4 = _mm256_permute2f128_ps(__tt0, __tt4, 0x31);
                        row5 = _mm256_permute2f128_ps(__tt1, __tt5, 0x31);
                        row6 = _mm256_permute2f128_ps(__tt2, __tt6, 0x31);
                        row7 = _mm256_permute2f128_ps(__tt3, __tt7, 0x31);

                        _mm256_storeu_ps(&dst[(j + tj) * n + (i + ti)], row0);
                        _mm256_storeu_ps(&dst[(j + tj + 1) * n + (i + ti)], row1);
                        _mm256_storeu_ps(&dst[(j + tj + 2) * n + (i + ti)], row2);
                        _mm256_storeu_ps(&dst[(j + tj + 3) * n + (i + ti)], row3);
                        _mm256_storeu_ps(&dst[(j + tj + 4) * n + (i + ti)], row4);
                        _mm256_storeu_ps(&dst[(j + tj + 5) * n + (i + ti)], row5);
                        _mm256_storeu_ps(&dst[(j + tj + 6) * n + (i + ti)], row6);
                        _mm256_storeu_ps(&dst[(j + tj + 7) * n + (i + ti)], row7);
                    }
                }
            }
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
        int m = n / 2;
        std::vector<float> src(n * m);
        std::vector<float> dst_standard(n * m);
        std::vector<float> dst_avx2_128bit(n * m);
        std::vector<float> dst_avx2_256bit_kernel1(n * m);
        std::vector<float> dst_avx2_256bit_kernel2(n * m);
        std::vector<float> dst_avx2_256bit_kernel3(n * m);

        for (float& val : src) {
            val = dis(gen);
        }

        if (PROFILING) {
            utils::compare_performance(
                "Standard transpose", 
                [&](){ transpose_nxm_baseline(src.data(), dst_standard.data(), n, m); },
                "AVX2 128-bit transpose", 
                [&](){ transpose_nxm_avx2_128bit(src.data(), dst_avx2_128bit.data(), n, m); },
                100
            );

            utils::compare_performance(
                "Standard transpose", 
                [&](){ transpose_nxm_baseline(src.data(), dst_standard.data(), n, m); },
                "AVX2 256-bit transpose kernel1", 
                [&](){ transpose_nxm_avx2_256bit_kernel1(src.data(), dst_avx2_256bit_kernel1.data(), n, m); },
                100
            );

            utils::compare_performance(
                "Standard transpose", 
                [&](){ transpose_nxm_baseline(src.data(), dst_standard.data(), n, m); },
                "AVX2 256-bit transpose kernel2", 
                [&](){ transpose_nxm_avx2_256bit_kernel2(src.data(), dst_avx2_256bit_kernel2.data(), n, m); },
                100
            );

            utils::compare_performance(
                "Standard transpose", 
                [&](){ transpose_nxm_baseline(src.data(), dst_standard.data(), n, m); },
                "AVX2 256-bit transpose kernel3", 
                [&](){ transpose_nxm_avx2_256bit_kernel3(src.data(), dst_avx2_256bit_kernel3.data(), n, m); },
                100
            );
        }

        bool is_correct_128bit = utils::check_correctness(dst_standard, dst_avx2_128bit);
        bool is_correct_256bit_kernel1 = utils::check_correctness(dst_standard, dst_avx2_256bit_kernel1);
        bool is_correct_256bit_kernel2 = utils::check_correctness(dst_standard, dst_avx2_256bit_kernel2);
        bool is_correct_256bit_kernel3 = utils::check_correctness(dst_standard, dst_avx2_256bit_kernel3);

        if (DEBUG) {
            std::cout << "Correctness check (128-bit AVX2): " << (is_correct_128bit ? "PASSED" : "FAILED") << std::endl;
            std::cout << "Correctness check (256-bit AVX2 kernel1): " << (is_correct_256bit_kernel1 ? "PASSED" : "FAILED") << std::endl;
            std::cout << "Correctness check (256-bit AVX2 kernel2): " << (is_correct_256bit_kernel2 ? "PASSED" : "FAILED") << std::endl;
            std::cout << "Correctness check (256-bit AVX2 kernel3): " << (is_correct_256bit_kernel3 ? "PASSED" : "FAILED") << std::endl;
        }

        if (!is_correct_128bit || !is_correct_256bit_kernel1 || !is_correct_256bit_kernel2 || !is_correct_256bit_kernel3) return EXIT_FAILURE;
    }

    return 0;
}
