#!/usr/bin/env bash
set -euo pipefail
artifact_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
driver_dir=$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
probe_binary="/tmp/dgx-gpu-first-touch-uvm"
nvcc -O2 -arch=sm_121 -DENABLE_UVM_TOOLS \
    -I"${driver_dir}/kernel-open/nvidia-uvm" \
    -I"${driver_dir}/kernel-open/common/inc" \
    "${artifact_dir}/gpu-first-touch.cu" -o "${probe_binary}" -ldl
"${probe_binary}" "${artifact_dir}/uvm-events.csv" 2>&1 | tee "${artifact_dir}/uvm-events-run.log"
