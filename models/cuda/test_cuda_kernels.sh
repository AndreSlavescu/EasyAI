#!/bin/bash
set -e

CU_DIR="$(dirname "$0")/search"
OUTPUT_EXEC="test_cuda_kernels"

CU_FILES=("$CU_DIR"/*.cu)
if [ ! -e "${CU_FILES[0]}" ]; then
    echo "No .cu files found in the directory: $CU_DIR"
    exit 1
fi

for CU_FILE in "${CU_FILES[@]}"; do
    KERNEL_NAME=$(basename "$CU_FILE" .cu)
    echo "Compiling CUDA kernel: $KERNEL_NAME"
    nvcc -std=c++17 -O3 -I"$CU_DIR" -o "$OUTPUT_EXEC" "$CU_FILE"

    if [ $? -ne 0 ]; then
        echo "Compilation of $KERNEL_NAME failed."
        exit 1
    fi

    echo "Compilation of $KERNEL_NAME successful. Running tests..."
    ./"$OUTPUT_EXEC"

    if [ $? -eq 0 ]; then
        echo "All tests for $KERNEL_NAME passed successfully."
    else
        echo "Some tests for $KERNEL_NAME failed."
        exit 1
    fi
done