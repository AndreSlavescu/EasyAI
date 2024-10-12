#pragma once

#include <chrono>
#include <functional>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>

namespace utils {

template<typename Func, typename... Args>
double measure_performance(
    int num_runs, Func&& func, 
    Args&&... args
) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_runs; ++i) {
        func(std::forward<Args>(args)...);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return static_cast<double>(duration.count()) / num_runs;
}

bool is_equal(float a, float b, float epsilon = 1e-5) {
    return fabs(a - b) < epsilon;
}

template<typename T>
bool check_correctness(
    const std::vector<T>& expected, 
    const std::vector<T>& actual, 
    T epsilon = std::numeric_limits<T>::epsilon() * 100
) {
    if (expected.size() != actual.size()) {
        std::cerr << "Size mismatch: expected " << expected.size() << ", got " << actual.size() << std::endl;
        return false;
    }

    for (size_t i = 0; i < expected.size(); ++i) {
        if (!is_equal(expected[i], actual[i], epsilon=epsilon)) {
            std::cerr << "Mismatch at index " << i << ": expected " << expected[i] << ", got " << actual[i] << std::endl;
            return false;
        }
    }
    return true;
}

template<typename Func1, typename Func2, typename... Args>
void compare_performance(
    const std::string& name1, 
    Func1&& func1, 
    const std::string& name2, 
    Func2&& func2, 
    int num_runs, 
    Args&&... args
) {
    double time1 = measure_performance(num_runs, func1, args...);
    double time2 = measure_performance(num_runs, func2, args...);

    std::cout << name1 << " average time: " << time1 << " microseconds" << std::endl;
    std::cout << name2 << " average time: " << time2 << " microseconds" << std::endl;
    std::cout << "Speedup: " << time1 / time2 << "x" << std::endl;
}

} // namespace utils
