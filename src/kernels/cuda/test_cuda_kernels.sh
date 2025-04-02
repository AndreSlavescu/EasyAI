#!/bin/bash
set -e

SCRIPT_DIR="$(dirname "$0")"
SEARCH_DIR="$SCRIPT_DIR/search"
TEMPLATED_DIR="$SCRIPT_DIR/templated_kernels"
OUTPUT_EXEC="test_cuda_kernels"

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo "Testing CUDA kernels..."

test_kernels_in_dir() {
    local dir="$1"
    local dir_name="$(basename "$dir")"
    
    echo -e "${GREEN}Testing kernels in $dir_name directory${NC}"
    
    CU_FILES=("$dir"/*.cu)
    if [ ! -e "${CU_FILES[0]}" ]; then
        echo "No .cu files found in the directory: $dir"
        return 0
    fi
    
    for CU_FILE in "${CU_FILES[@]}"; do
        KERNEL_NAME=$(basename "$CU_FILE" .cu)
        echo "Compiling CUDA kernel: $KERNEL_NAME"
        nvcc -std=c++17 -I"$SCRIPT_DIR" -o "$OUTPUT_EXEC" "$CU_FILE" -lcublas
        
        if [ $? -ne 0 ]; then
            echo -e "${RED}Compilation of $KERNEL_NAME failed.${NC}"
            exit 1
        fi
        
        echo "Compilation of $KERNEL_NAME successful. Running tests..."
        ./"$OUTPUT_EXEC"
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}All tests for $KERNEL_NAME passed successfully.${NC}"
        else
            echo -e "${RED}Some tests for $KERNEL_NAME failed.${NC}"
            exit 1
        fi
    done
}

test_kernels_in_dir "$SEARCH_DIR"
test_kernels_in_dir "$TEMPLATED_DIR"

if [ -e "$OUTPUT_EXEC" ]; then
    rm "$OUTPUT_EXEC"
fi

echo -e "${GREEN}All CUDA kernel tests completed successfully.${NC}"