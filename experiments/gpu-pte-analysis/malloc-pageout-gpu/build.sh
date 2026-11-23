#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
driver_dir=$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
/usr/local/cuda/bin/nvcc -O2 -arch=sm_121 \
    -I"$driver_dir/kernel-open/nvidia-uvm" \
    -I"$driver_dir/kernel-open/common/inc" \
    malloc_pageout.cu -o malloc_pageout
