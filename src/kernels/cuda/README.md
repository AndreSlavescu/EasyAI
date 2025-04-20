# CUDA Implementations of popular Models and Algorithms

## Sections

### Search Kernels

Search kernels can be found under the [`search`](search) directory.

### Tiling Template Library + Kernels

Kernels using the custom tiling library can be found under the [`templated_tiling_kernels`](templated_tiling_kernels) directory.

### Tensor Core Kernels

Kernels using tensor core instructions for ampere and earlier can be found under the [`tensor_core_kernels`](tensor_core_kernels) directory.

### PTX Kernels

Low-level PTX kernels can be found under the [`ptx_kernels`](ptx_kernels) directory. These kernels are written using hand-optimized PTX, with only the necessary CUDA driver code to launch the kernels.

### Simplified CUDA and PTX Strategies

Documentation around common optimization strategies, and particular keywords in CUDA + PTX, can be found under the [`docs`](docs) directory.