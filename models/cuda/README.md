# CUDA Implementations of popular Models and Algorithms

## '__restrict__' Keyword

CUDA shares the same semantics as C99 for general use of __restrict__,
which is a hint to the compiler that the pointers are not aliased.
On NVidia GPUs with Compute Capability 3.5 (Kepler) and higher,
__restrict__ also suggests to the compiler that global memory loads
should use the read-only cache.
See the following link for more details:
https://developer.nvidia.com/blog/cuda-pro-tip-optimize-pointer-aliasing/
Also see the following link for __restrict__ definition in the CUDA C/C++ Programming Guide:
https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#restrict