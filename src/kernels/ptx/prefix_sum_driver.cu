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

void cpu_prefix_sum(float *output, int N, float val) {
    for (int i = 0; i < N; i++) {
        output[i] = val * (i + 1);
    }
}

int main() {
    CUDA_CHECK(cuInit(0));
    CUdevice cuDevice;
    CUDA_CHECK(cuDeviceGet(&cuDevice, 0));
    CUcontext cuContext;
    CUDA_CHECK(cuCtxCreate(&cuContext, 0, cuDevice));

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

    CUmodule cuModule;
    CUDA_CHECK(cuModuleLoadData(&cuModule, ptx_content.c_str()));

    CUfunction kernel;
    CUDA_CHECK(cuModuleGetFunction(&kernel, cuModule, "warpPrefixSum"));

    CUdeviceptr d_output;
    size_t output_size = WARP_SIZE * sizeof(float);
    CUDA_CHECK(cuMemAlloc(&d_output, output_size));

    float inputVal = 1.0f;
    void *kernel_params[2];
    kernel_params[0] = &inputVal;
    kernel_params[1] = &d_output;

    CUDA_CHECK(cuLaunchKernel(kernel,
                              1, 1, 1,
                              WARP_SIZE, 1, 1,
                              0,
                              0,
                              kernel_params,
                              0));
    CUDA_CHECK(cuCtxSynchronize());

    float h_output[WARP_SIZE] = {0};
    CUDA_CHECK(cuMemcpyDtoH(h_output, d_output, output_size));

    float expected[WARP_SIZE] = {0};
    cpu_prefix_sum(expected, WARP_SIZE, inputVal);

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

    printf("\nWarp prefix sum results (GPU):\n");
    for (int i = 0; i < WARP_SIZE; i++) {
        printf("Thread %2d: %f\n", i, h_output[i]);
    }

    CUDA_CHECK(cuMemFree(d_output));
    CUDA_CHECK(cuModuleUnload(cuModule));
    CUDA_CHECK(cuCtxDestroy(cuContext));

    return 0;
}
