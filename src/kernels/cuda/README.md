# CUDA Implementations of popular Models and Algorithms

## Sections

### Search Kernels

Search kernels can be found under the [`src/kernels/cuda/search`](src/kernels/cuda/search) directory.

### Tiling Template Library + Kernels

Kernels using the custom tiling library can be found under the [`src/kernels/cuda/templated_tiling_kernels`](src/kernels/cuda/templated_tiling_kernels) directory.

### Tensor Core Kernels

Kernels using tensor core instructions for ampere and earlier can be found under the [`src/kernels/cuda/tensor_core_kernels`](src/kernels/cuda/tensor_core_kernels) directory.

### PTX Kernels

Low-level PTX kernels can be found under the [`src/kernels/cuda/ptx_kernels`](src/kernels/cuda/ptx_kernels) directory. These kernels are written using hand-optimized PTX, with only the necessary CUDA driver code to launch the kernels.

### Simplified CUDA and PTX Strategies

Documentation around common optimization strategies, and particular keywords in CUDA + PTX, can be found under the [`src/kernels/cuda/docs`](src/kernels/cuda/docs) directory.