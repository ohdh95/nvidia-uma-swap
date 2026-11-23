#!/usr/bin/env bash
set -euo pipefail
vmm_check_dir=$(cd -- "$(dirname -- "$0")" && pwd)
/usr/local/cuda/bin/nvcc -std=c++17 -arch=sm_121 "$vmm_check_dir/cpu-gpu-folio.cu" -lcuda -o /tmp/dgx-vmm-host-folio-probe
/tmp/dgx-vmm-host-folio-probe
