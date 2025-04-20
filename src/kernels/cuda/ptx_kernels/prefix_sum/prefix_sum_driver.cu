#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string>
#include <cstdio>
#include <cuda_fp16.h>
#include "../../cuda_utils.h"

void cpu_prefix_sum(float *input, float *output, int N) {
    output[0] = input[0];
    for (int i = 1; i < N; ++i) {
        output[i] = output[i - 1] + input[i];
    }
}

int main() {
    CHECK_CUDA_RESULT(cuInit(0));
    CUdevice cuDevice;
    CHECK_CUDA_RESULT(cuDeviceGet(&cuDevice, 0));
    CUcontext cuContext;
    CHECK_CUDA_RESULT(cuCtxCreate(&cuContext, 0, cuDevice));

    std::string ptx_path = "prefix_sum/prefix_sum.ptx";
    FILE* ptx_file = fopen(ptx_path.c_str(), "rb");
    if (!ptx_file) {
        printf("Failed to open PTX file: %s\n", ptx_path.c_str());
        return 1;
    }
    printf("Successfully opened PTX file: %s\n", ptx_path.c_str());

    fseek(ptx_file, 0, SEEK_END);
    size_t size = ftell(ptx_file);
    fseek(ptx_file, 0, SEEK_SET);

    std::string ptx_content;
    ptx_content.resize(size);
    fread(&ptx_content[0], 1, size, ptx_file);
    fclose(ptx_file);

    char error_log[8192];
    char info_log [8192];
    size_t logSize = sizeof(error_log);
    CUjit_option options[4] = {
        CU_JIT_ERROR_LOG_BUFFER,
        CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_INFO_LOG_BUFFER,
        CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES
    };
    void* values[4] = {
        (void*)error_log,
        (void*)logSize,
        (void*)info_log,
        (void*)logSize
    };
    memset(error_log, 0, logSize);
    memset(info_log , 0, logSize);

    CUmodule cuModule;
    CUresult result = cuModuleLoadDataEx(
        &cuModule,
        ptx_content.c_str(),
        4,
        options,
        values
    );
    if (result != CUDA_SUCCESS) {
        const char* error_string;
        cuGetErrorString(result, &error_string);
        printf("cuModuleLoadDataEx failed with error: %s (code %d)\n",
               error_string, static_cast<int>(result));
        printf("%s\n", error_log);
        printf("%s\n", info_log);
        return 1;
    }

    CUfunction kernel;
    CHECK_CUDA_RESULT(cuModuleGetFunction(&kernel, cuModule, "warpPrefixSum"));

    float h_input[WARP_SIZE];
    for (int i = 0; i < WARP_SIZE; ++i) {
        h_input[i] = static_cast<float>(i + 1);
    }

    CUdeviceptr d_input, d_output;
    size_t array_size = WARP_SIZE * sizeof(float);
    CHECK_CUDA_RESULT(cuMemAlloc(&d_input, array_size));
    CHECK_CUDA_RESULT(cuMemAlloc(&d_output, array_size));
    CHECK_CUDA_RESULT(cuMemcpyHtoD(d_input, h_input, array_size));

    void* kernel_params[] = { &d_input, &d_output };
    CHECK_CUDA_RESULT(cuLaunchKernel(
        kernel,
        1, 1, 1,
        WARP_SIZE, 1, 1,
        0,
        0,
        kernel_params,
        0
    ));
    CHECK_CUDA_RESULT(cuCtxSynchronize());

    float h_output[WARP_SIZE];
    CHECK_CUDA_RESULT(cuMemcpyDtoH(h_output, d_output, array_size));

    float expected[WARP_SIZE];
    cpu_prefix_sum(h_input, expected, WARP_SIZE);

    int correct = 1;
    for (int i = 0; i < WARP_SIZE; ++i) {
        if (fabs(h_output[i] - expected[i]) > 1e-6f) {
            correct = 0;
            printf("Mismatch at index %d: GPU = %f, CPU = %f\n",
                   i, h_output[i], expected[i]);
        }
    }
    printf(correct ? "Test PASSED! GPU results match CPU results.\n"
                   : "Test FAILED!\n");

    printf("\nWarp cumulative sum results (GPU):\n");
    for (int i = 0; i < WARP_SIZE; ++i) {
        printf("Element %2d: %f\n", i, h_output[i]);
    }

    CHECK_CUDA_RESULT(cuMemFree(d_input));
    CHECK_CUDA_RESULT(cuMemFree(d_output));
    CHECK_CUDA_RESULT(cuModuleUnload(cuModule));
    CHECK_CUDA_RESULT(cuCtxDestroy(cuContext));

    return 0;
}