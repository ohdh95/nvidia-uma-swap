#!/usr/bin/env bash
set -euo pipefail
managed_probe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
managed_driver_dir=$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
managed_run_dir=$(mktemp -d "$managed_probe_dir/repeat-XXXXXX")
/usr/local/cuda/bin/nvcc -O2 -arch=sm_121 -DENABLE_UVM_TOOLS \
    -I"$managed_driver_dir/kernel-open/nvidia-uvm" \
    -I"$managed_driver_dir/kernel-open/common/inc" \
    "$managed_probe_dir/gpu-first-cpu-pte.cu" -o "$managed_run_dir/probe" -ldl
"$managed_run_dir/probe" "$managed_run_dir/uvm-events.csv" \
    >"$managed_run_dir/run.stdout" 2>"$managed_run_dir/run.log"
cat "$managed_run_dir/run.log"
python3 "$managed_probe_dir/analyze.py" "$managed_run_dir"
printf 'Results: %s\n' "$managed_run_dir"
