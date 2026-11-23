#!/usr/bin/env bash
set -euo pipefail
gpu_cpu_pte_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
gpu_cpu_pte_driver_dir=$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
/usr/local/cuda/bin/nvcc -O2 -arch=sm_121 -DENABLE_UVM_TOOLS \
    -I"$gpu_cpu_pte_driver_dir/kernel-open/nvidia-uvm" \
    -I"$gpu_cpu_pte_driver_dir/kernel-open/common/inc" \
    "$gpu_cpu_pte_dir/gpu-first-cpu-pte.cu" -o /tmp/dgx-gpu-first-cpu-pte-probe -ldl
/tmp/dgx-gpu-first-cpu-pte-probe "$gpu_cpu_pte_dir/uvm-events.csv" 2>&1 | tee "$gpu_cpu_pte_dir/run.log"
