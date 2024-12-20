#pragma once

#include <math.h>
#include <stddef.h>
#include <string.h>
#include <omp.h>

void relu_act(void* input, void* output, size_t size, size_t elem_size) {
    float* in = (float*)input;
    float* out = (float*)output;
    
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        out[i] = in[i] > 0 ? in[i] : 0;
    }
}

void tanh_act(void* input, void* output, size_t size, size_t elem_size) {
    float* in = (float*)input;
    float* out = (float*)output;
    
    for (size_t i = 0; i < size; i++) {
        out[i] = tanh(in[i]);
    }
}

void batchnorm2d(void* input, void* output, size_t size, size_t elem_size) {
    float* in = (float*)input;
    float* out = (float*)output;
    
    float mean = 0.0f;
    #pragma omp parallel for reduction(+:mean)
    for (size_t i = 0; i < size; i++) {
        mean += in[i];
    }
    mean /= size;

    float var = 0.0f;
    #pragma omp parallel for reduction(+:var)
    for (size_t i = 0; i < size; i++) {
        var += (in[i] - mean) * (in[i] - mean);
    }
    var /= size;

    float eps = 1e-5f;
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        out[i] = (in[i] - mean) / sqrt(var + eps);
    }
}

void deconv2d(void* input, void* output, size_t in_height, size_t in_width, size_t kernel_size, size_t stride) {
    float* in = (float*)input;
    float* out = (float*)output;
    
    size_t out_height = (in_height - 1) * stride + kernel_size;
    size_t out_width = (in_width - 1) * stride + kernel_size;
    
    memset(out, 0, out_height * out_width * sizeof(float));

    for (size_t i = 0; i < in_height; i++) {
        for (size_t j = 0; j < in_width; j++) {
            size_t out_i = i * stride;
            size_t out_j = j * stride;
            
            for (size_t ki = 0; ki < kernel_size; ki++) {
                for (size_t kj = 0; kj < kernel_size; kj++) {
                    if (out_i + ki < out_height && out_j + kj < out_width) {
                        out[(out_i + ki) * out_width + (out_j + kj)] = in[i * in_width + j];
                    }
                }
            }
        }
    }
}
