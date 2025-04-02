#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string>
#include <cstdio>

#ifndef PTX_DIR
#define PTX_DIR "."
#endif

#define CUDA_CHECK(call)                                                                \
    do {                                                                                \
        CUresult res = (call);                                                          \
        if (res != CUDA_SUCCESS) {                                                      \
            const char *errStr;                                                         \
            cuGetErrorString(res, &errStr);                                             \
            fprintf(stderr, "CUDA error at %s:%d -- %s\n", __FILE__, __LINE__, errStr); \
            exit(EXIT_FAILURE);                                                         \
        }                                                                               \
    } while (0)

#define WARP_SIZE 32

// This function computes the cumulative sum on CPU.
void cpu_prefix_sum(float *input, float *output, int N) {
    output[0] = input[0];
    for (int i = 1; i < N; i++) {
        output[i] = output[i-1] + input[i];
    }
}

int main() {
    // Initialize CUDA driver API and create context
    CUDA_CHECK(cuInit(0));
    CUdevice cuDevice;
    CUDA_CHECK(cuDeviceGet(&cuDevice, 0));
    CUcontext cuContext;
    CUDA_CHECK(cuCtxCreate(&cuContext, 0, cuDevice));

    // Load the PTX file containing the cumulative sum (prefix sum) kernel
    std::string ptx_path = std::string(PTX_DIR) + "/prefix_sum.ptx";
    FILE* ptx_file = fopen(ptx_path.c_str(), "rb");
    if (!ptx_file) {
        printf("Failed to open PTX file: %s\n", ptx_path.c_str());
        return 1;
    }
    
    fseek(ptx_file, 0, SEEK_END);
    size_t size = ftell(ptx_file);
    fseek(ptx_file, 0, SEEK_SET);
    
    std::string ptx_content;
    ptx_content.resize(size);
    fread(&ptx_content[0], 1, size, ptx_file);
    fclose(ptx_file);

    // Load module and get kernel function "warpPrefixSum"
    CUmodule cuModule;
    CUDA_CHECK(cuModuleLoadData(&cuModule, ptx_content.c_str()));

    CUfunction kernel;
    // Assuming our PTX kernel "warpPrefixSum" now calculates cumulative sum on an input array.
    CUDA_CHECK(cuModuleGetFunction(&kernel, cuModule, "warpPrefixSum"));

    // Create and initialize the host input array
    float h_input[WARP_SIZE];
    for (int i = 0; i < WARP_SIZE; i++) {
        h_input[i] = static_cast<float>(i + 1); // Example values: 1.0, 2.0, ..., 32.0
    }

    // Allocate device memory for input and output arrays
    CUdeviceptr d_input, d_output;
    size_t array_size = WARP_SIZE * sizeof(float);
    CUDA_CHECK(cuMemAlloc(&d_input, array_size));
    CUDA_CHECK(cuMemAlloc(&d_output, array_size));

    // Copy the host input array to the device
    CUDA_CHECK(cuMemcpyHtoD(d_input, h_input, array_size));

    // Set up kernel parameters: our kernel now takes two pointers (input and output)
    void *kernel_params[2];
    kernel_params[0] = &d_input;
    kernel_params[1] = &d_output;

    // Launch the kernel with one block of WARP_SIZE threads
    CUDA_CHECK(cuLaunchKernel(kernel,
                              1, 1, 1,            // grid dimensions
                              WARP_SIZE, 1, 1,      // block dimensions
                              0,                    // shared memory
                              0,                    // stream
                              kernel_params,
                              0));
    CUDA_CHECK(cuCtxSynchronize());

    // Retrieve the output from device memory to the host
    float h_output[WARP_SIZE] = {0};
    CUDA_CHECK(cuMemcpyDtoH(h_output, d_output, array_size));

    // Compute the expected cumulative sum on the CPU for comparison
    float expected[WARP_SIZE] = {0};
    cpu_prefix_sum(h_input, expected, WARP_SIZE);

    // Check for correctness
    int correct = 1;
    for (int i = 0; i < WARP_SIZE; i++) {
        if (fabs(h_output[i] - expected[i]) > 1e-6) {
            correct = 0;
            printf("Mismatch at index %d: GPU = %f, CPU = %f\n", i, h_output[i], expected[i]);
        }
    }
    if (correct) {
        printf("Test PASSED! GPU results match CPU results.\n");
    } else {
        printf("Test FAILED!\n");
    }

    // Print the cumulative sum results from the GPU
    printf("\nWarp cumulative sum results (GPU):\n");
    for (int i = 0; i < WARP_SIZE; i++) {
        printf("Element %2d: %f\n", i, h_output[i]);
    }

    // Free device memory and clean up
    CUDA_CHECK(cuMemFree(d_input));
    CUDA_CHECK(cuMemFree(d_output));
    CUDA_CHECK(cuModuleUnload(cuModule));
    CUDA_CHECK(cuCtxDestroy(cuContext));

    return 0;
}
