#include <iostream>
#include <arm_neon.h>
#include <vector>
#include <thread>
#include <algorithm>
#include <chrono>
#include <random>
#include <functional>

#define DEBUG 0
#define PROFILING 1

void radix_sort_baseline(uint32_t* arr, size_t n) {
    std::vector<uint32_t> temp(n);
    uint32_t* input = arr;
    uint32_t* output = temp.data();

    for (int shift = 0; shift < 32; shift += 8) {
        uint32_t count[256] = {0};
        
        for (size_t i = 0; i < n; i++) {
            count[(input[i] >> shift) & 0xFF]++;
        }
        
        uint32_t total = 0;
        for (int i = 0; i < 256; i++) {
            uint32_t old_count = count[i];
            count[i] = total;
            total += old_count;
        }
        
        for (size_t i = 0; i < n; i++) {
            uint32_t digit = (input[i] >> shift) & 0xFF;
            output[count[digit]++] = input[i];
        }
        
        std::swap(input, output);
    }
    
    if (input != arr) {
        std::copy(temp.begin(), temp.end(), arr);
    }
}

void countSort(const uint32_t* input, uint32_t* output, size_t n, int shift) {
    uint32_t histogram[256] = {0};
    
    size_t vec_size = n - (n % 4);
    for (size_t i = 0; i < vec_size; i += 4) {
        uint32x4_t data = vld1q_u32(&input[i]);
        
        uint32x4_t shift_vec = vdupq_n_u32(shift);
        uint32x4_t shifted = vshlq_u32(data, vnegq_s32(vreinterpretq_s32_u32(shift_vec)));
        uint32x4_t masked = vandq_u32(shifted, vdupq_n_u32(0xFF));
        
        uint32_t bytes[4];
        vst1q_u32(bytes, masked);
        histogram[bytes[0]]++;
        histogram[bytes[1]]++;
        histogram[bytes[2]]++;
        histogram[bytes[3]]++;
    }
    
    for (size_t i = vec_size; i < n; i++) {
        histogram[(input[i] >> shift) & 0xFF]++;
    }
    
    uint32_t sum = 0;
    for (int i = 0; i < 256; i++) {
        uint32_t temp = histogram[i];
        histogram[i] = sum;
        sum += temp;
    }
    
    for (size_t i = 0; i < n; i++) {
        uint32_t byte = (input[i] >> shift) & 0xFF;
        output[histogram[byte]++] = input[i];
    }
}

void parallelRadixSort(uint32_t* arr, size_t n, int num_threads = 4) {
    std::vector<uint32_t> temp(n);
    uint32_t* input = arr;
    uint32_t* output = temp.data();
    
    for (int shift = 0; shift < 32; shift += 8) {
        countSort(input, output, n, shift);
        std::swap(input, output);
    }
    
    if (input != arr) {
        std::copy(temp.begin(), temp.end(), arr);
    }
}

namespace utils {
    template<typename Func>
    double measure_performance(int num_runs, Func&& func) {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_runs; ++i) {
            func();
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return static_cast<double>(duration.count()) / num_runs;
    }

    template<typename Func1, typename Func2>
    void compare_performance(
        const std::string& name1,
        Func1&& func1,
        const std::string& name2,
        Func2&& func2,
        int num_runs
    ) {
        double time1 = measure_performance(num_runs, func1);
        double time2 = measure_performance(num_runs, func2);

        std::cout << name1 << " average time: " << time1 << " microseconds" << std::endl;
        std::cout << name2 << " average time: " << time2 << " microseconds" << std::endl;
        std::cout << "Speedup: " << time1 / time2 << "x" << std::endl;
    }

    bool check_correctness(const std::vector<uint32_t>& expected, const std::vector<uint32_t>& actual) {
        if (expected.size() != actual.size()) {
            std::cerr << "Size mismatch: expected " << expected.size() << ", got " << actual.size() << std::endl;
            return false;
        }

        for (size_t i = 0; i < expected.size(); ++i) {
            if (expected[i] != actual[i]) {
                std::cerr << "Mismatch at index " << i << ": expected " << expected[i] << ", got " << actual[i] << std::endl;
                return false;
            }
        }
        return true;
    }
}

int main() {
    std::vector<int> test_sizes = {1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18};
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dis(0, UINT32_MAX);

    for (const int n : test_sizes) {
        std::cout << "\nTesting size n = " << n << std::endl;
        
        std::vector<uint32_t> data(n);
        std::vector<uint32_t> data_baseline(n);
        std::vector<uint32_t> data_neon(n);

        // Initialize with random values
        for (size_t i = 0; i < n; i++) {
            data[i] = dis(gen);
        }

        // Copy data for each implementation
        data_baseline = data;
        data_neon = data;

        if (PROFILING) {
            utils::compare_performance(
                "Baseline radix sort",
                [&](){ radix_sort_baseline(data_baseline.data(), n); },
                "NEON radix sort",
                [&](){ parallelRadixSort(data_neon.data(), n); },
                100
            );
        }

        // Verify correctness
        std::vector<uint32_t> sorted_reference = data;
        std::sort(sorted_reference.begin(), sorted_reference.end());

        bool baseline_correct = utils::check_correctness(sorted_reference, data_baseline);
        bool neon_correct = utils::check_correctness(sorted_reference, data_neon);

        if (DEBUG) {
            std::cout << "Baseline sort: " << (baseline_correct ? "PASSED" : "FAILED") << std::endl;
            std::cout << "NEON sort: " << (neon_correct ? "PASSED" : "FAILED") << std::endl;
        }

        if (!baseline_correct || !neon_correct) {
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}