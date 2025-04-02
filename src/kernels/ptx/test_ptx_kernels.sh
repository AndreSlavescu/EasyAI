#!/bin/bash
KERNEL_DIR="$(dirname "$0")"
PTX_OUTPUT_DIR="$KERNEL_DIR/compiled"
OUTPUT_EXEC="test_ptx_kernel"

mkdir -p "$PTX_OUTPUT_DIR"

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

mapfile -t CUDA_FILES < <(find "$KERNEL_DIR" -type f -name "*.cu")
if [ ${#CUDA_FILES[@]} -eq 0 ]; then
    echo "No .cu files found in the directory: $KERNEL_DIR"
    exit 1
fi

echo "Starting PTX kernel compilation and testing..."

for cuda_file in "${CUDA_FILES[@]}"; do
    filename=$(basename "$cuda_file")
    kernel_name="${filename%.cu}"
    ptx_base="${kernel_name%_driver}"
    ptx_name="${ptx_base}.ptx"
    ptx_dir=$(dirname "$cuda_file")
    
    cp "$ptx_dir/$ptx_name" "$PTX_OUTPUT_DIR/$ptx_name"
    
    echo "Using PTX file: $ptx_name"
    
    if ! ptxas -arch=sm_80 "$PTX_OUTPUT_DIR/$ptx_name" -o /dev/null; then
        echo -e "${RED}PTX validation failed for $ptx_name${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}PTX validation successful${NC}"
    
    if ! nvcc -o "$OUTPUT_EXEC" "$cuda_file" -I"$KERNEL_DIR" -lcuda -DPTX_DIR="\"$PTX_OUTPUT_DIR\""; then
        echo -e "${RED}Test compilation failed for $kernel_name${NC}"
        exit 1
    fi
    
    if ./"$OUTPUT_EXEC"; then
        echo -e "${GREEN}All tests for $kernel_name passed successfully.${NC}"
    else
        echo -e "${RED}Some tests for $kernel_name failed.${NC}"
        exit 1
    fi
done

if [ -e "$OUTPUT_EXEC" ]; then
    rm "$OUTPUT_EXEC"
fi

if [ -d "$PTX_OUTPUT_DIR" ]; then
    rm -rf "$PTX_OUTPUT_DIR"
fi

echo "All kernel compilations and tests completed successfully."