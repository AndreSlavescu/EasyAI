#include <iostream>
#include <arm_neon.h>
#include <vector>
#include <thread>
#include <algorithm>
#include <chrono>
#include <random>
#include <functional>
#include <iomanip>

#define DEBUG 0
#define PROFILING 0

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

void countSortSmall(const uint32_t* input, uint32_t* output, size_t n, int shift) {
    constexpr int HISTOGRAM_SIZE = 256;
    uint32_t histogram[HISTOGRAM_SIZE] = {0};
    
    size_t vec_size = n - (n % 16);
    for (size_t block = 0; block < vec_size; block += 16) {
        uint32x4_t data1 = vld1q_u32(&input[block]);
        uint32x4_t data2 = vld1q_u32(&input[block + 4]);
        uint32x4_t data3 = vld1q_u32(&input[block + 8]);
        uint32x4_t data4 = vld1q_u32(&input[block + 12]);
        
        int32x4_t shift_vec = vdupq_n_s32(-shift);
        uint32x4_t masked1 = vandq_u32(vshlq_u32(data1, shift_vec), vdupq_n_u32(0xFF));
        uint32x4_t masked2 = vandq_u32(vshlq_u32(data2, shift_vec), vdupq_n_u32(0xFF));
        uint32x4_t masked3 = vandq_u32(vshlq_u32(data3, shift_vec), vdupq_n_u32(0xFF));
        uint32x4_t masked4 = vandq_u32(vshlq_u32(data4, shift_vec), vdupq_n_u32(0xFF));
        
        uint32_t bytes[16];
        vst1q_u32(bytes, masked1);
        vst1q_u32(bytes + 4, masked2);
        vst1q_u32(bytes + 8, masked3);
        vst1q_u32(bytes + 12, masked4);
        
        for (int j = 0; j < 16; j++) {
            histogram[bytes[j]]++;
        }
    }
    
    for (size_t i = vec_size; i < n; i++) {
        histogram[(input[i] >> shift) & 0xFF]++;
    }
    
    uint32_t sum = 0;
    for (int i = 0; i < HISTOGRAM_SIZE; i++) {
        uint32_t temp = histogram[i];
        histogram[i] = sum;
        sum += temp;
    }
    
    for (size_t i = 0; i < n; i++) {
        uint32_t byte = (input[i] >> shift) & 0xFF;
        output[histogram[byte]++] = input[i];
    }
}

void countSortLarge(const uint32_t* input, uint32_t* output, size_t n, int shift) {
    constexpr int HISTOGRAM_SIZE = 256;
    constexpr size_t MIN_BLOCK_SIZE = 1 << 20;
    
    int num_threads = std::min(static_cast<int>(n / MIN_BLOCK_SIZE + 1), 
                             static_cast<int>(std::thread::hardware_concurrency()));
    
    std::vector<std::array<uint32_t, HISTOGRAM_SIZE>> local_histograms(num_threads);
    std::vector<std::thread> threads;
    
    for (int t = 0; t < num_threads; t++) {
        size_t chunk_size = n / num_threads;
        size_t start = t * chunk_size;
        size_t end = (t == num_threads - 1) ? n : (t + 1) * chunk_size;
        
        threads.emplace_back([&, t, start, end]() {
            auto& my_histogram = local_histograms[t];
            std::fill(my_histogram.begin(), my_histogram.end(), 0);
            
            size_t vec_size = start + ((end - start) / 16) * 16;
            for (size_t block = start; block < vec_size; block += 16) {
                uint32x4_t data1 = vld1q_u32(&input[block]);
                uint32x4_t data2 = vld1q_u32(&input[block + 4]);
                uint32x4_t data3 = vld1q_u32(&input[block + 8]);
                uint32x4_t data4 = vld1q_u32(&input[block + 12]);
                
                int32x4_t shift_vec = vdupq_n_s32(-shift);
                uint32x4_t masked1 = vandq_u32(vshlq_u32(data1, shift_vec), vdupq_n_u32(0xFF));
                uint32x4_t masked2 = vandq_u32(vshlq_u32(data2, shift_vec), vdupq_n_u32(0xFF));
                uint32x4_t masked3 = vandq_u32(vshlq_u32(data3, shift_vec), vdupq_n_u32(0xFF));
                uint32x4_t masked4 = vandq_u32(vshlq_u32(data4, shift_vec), vdupq_n_u32(0xFF));
                
                uint32_t bytes[16];
                vst1q_u32(bytes, masked1);
                vst1q_u32(bytes + 4, masked2);
                vst1q_u32(bytes + 8, masked3);
                vst1q_u32(bytes + 12, masked4);
                
                for (int j = 0; j < 16; j++) {
                    my_histogram[bytes[j]]++;
                }
            }
            
            for (size_t i = vec_size; i < end; i++) {
                my_histogram[(input[i] >> shift) & 0xFF]++;
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    std::array<uint32_t, HISTOGRAM_SIZE> global_histogram = {0};
    for (const auto& hist : local_histograms) {
        for (int i = 0; i < HISTOGRAM_SIZE; i++) {
            global_histogram[i] += hist[i];
        }
    }
    
    uint32_t sum = 0;
    for (int i = 0; i < HISTOGRAM_SIZE; i++) {
        uint32_t temp = global_histogram[i];
        global_histogram[i] = sum;
        sum += temp;
    }
    
    std::vector<std::array<uint32_t, HISTOGRAM_SIZE>> thread_offsets(num_threads);
    thread_offsets[0] = global_histogram;
    
    for (int t = 1; t < num_threads; t++) {
        thread_offsets[t] = thread_offsets[t-1];
        for (int i = 0; i < HISTOGRAM_SIZE; i++) {
            thread_offsets[t][i] += local_histograms[t-1][i];
        }
    }
    
    threads.clear();
    for (int t = 0; t < num_threads; t++) {
        size_t chunk_size = n / num_threads;
        size_t start = t * chunk_size;
        size_t end = (t == num_threads - 1) ? n : (t + 1) * chunk_size;
        
        threads.emplace_back([&, t, start, end]() {
            auto& offsets = thread_offsets[t];
            for (size_t i = start; i < end; i++) {
                uint32_t byte = (input[i] >> shift) & 0xFF;
                output[offsets[byte]++] = input[i];
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
}

void countSort(const uint32_t* input, uint32_t* output, size_t n, int shift) {
    constexpr size_t SMALL_SIZE_THRESHOLD = 1 << 18;
    
    if (n < SMALL_SIZE_THRESHOLD) {
        countSortSmall(input, output, n, shift);
    } else {
        countSortLarge(input, output, n, shift);
    }
}

void parallelRadixSort(uint32_t* arr, size_t n) {
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
    std::vector<size_t> test_sizes = {
        1 << 10,    // 1K
        1 << 12,    // 4K
        1 << 14,    // 16K
        1 << 16,    // 64K
        1 << 18,    // 256K
        1 << 20,    // 1M
        1 << 22,    // 4M
        1 << 24,    // 16M
        1 << 26     // 64M
    };
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dis(0, UINT32_MAX);

    for (const size_t n : test_sizes) {
        #if PROFILING
        double size_mb = (n * sizeof(uint32_t)) / (1024.0 * 1024.0);
        std::cout << "\nTesting size n = " << n << " elements (" 
                  << std::fixed << std::setprecision(2) << size_mb << " MB)" << std::endl;
        #endif
        
        std::vector<uint32_t> data(n);
        #if DEBUG
        std::vector<uint32_t> data_baseline(n);
        #endif
        std::vector<uint32_t> data_neon(n);

        for (size_t i = 0; i < n; i++) {
            data[i] = dis(gen);
        }
        #if DEBUG
        data_baseline = data;
        #endif
        data_neon = data;

        #if DEBUG
        if (n <= (1 << 20)) { 
            auto start = std::chrono::high_resolution_clock::now();
            std::sort(data_baseline.begin(), data_baseline.end());
            auto end = std::chrono::high_resolution_clock::now();
            auto baseline_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            std::cout << "std::sort time: " << std::fixed << std::setprecision(2) 
                      << baseline_time / 1000.0 << " ms" << std::endl;
        }
        #endif

        int num_iterations = n <= (1 << 16) ? 100 : 1;
        #if PROFILING
        auto start = std::chrono::high_resolution_clock::now();
        #endif
        
        for (int i = 0; i < num_iterations; i++) {
            if (i > 0) data_neon = data;  
            parallelRadixSort(data_neon.data(), n);
        }
        
        #if PROFILING
        auto end = std::chrono::high_resolution_clock::now();
        auto radix_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        double avg_time_ms = (radix_time / static_cast<double>(num_iterations)) / 1000.0;
        std::cout << "Radix sort time: " << std::fixed << std::setprecision(2) 
                  << avg_time_ms << " ms" << std::endl;
        #endif
        
        bool correct = true;
        #if DEBUG
        if (n <= (1 << 20)) {
            correct = std::equal(data_baseline.begin(), data_baseline.end(), data_neon.begin());
        } else {
            correct = std::is_sorted(data_neon.begin(), data_neon.end());
        }
        
        std::cout << "Correctness: " << (correct ? "PASS" : "FAIL") << std::endl;
        #endif
        
        #if PROFILING
        double throughput = 0.0;
        if (avg_time_ms > 0.0) {
            throughput = (size_mb / 1024.0) / (avg_time_ms / 1000.0);
        }
        
        std::cout << "Throughput: " << std::fixed << std::setprecision(2);
        if (std::isinf(throughput)) {
            std::cout << ">1000" << " GB/s" << std::endl;
        } else {
            std::cout << throughput << " GB/s" << std::endl;
        }
        #endif
        
        if (!correct) {
            std::cerr << "Sort failed for size " << n << std::endl;
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}