#!/bin/bash
set -e

CPP_DIR="$(dirname "$0")/matrix_methods"
OUTPUT_EXEC="test_avx2_kernels"

CPP_FILES=("$CPP_DIR"/*.cpp)
if [ ! -e "${CPP_FILES[0]}" ]; then
    echo "No .cpp files found in the directory: $CPP_DIR"
    exit 1
fi

for CPP_FILE in "${CPP_FILES[@]}"; do
    KERNEL_NAME=$(basename "$CPP_FILE" .cpp)
    echo "Compiling C++ file: $KERNEL_NAME"
    g++ -mavx2 -std=c++17 -I"$CPP_DIR" -o "$OUTPUT_EXEC" "$CPP_FILE"

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
