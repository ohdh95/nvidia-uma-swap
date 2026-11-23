#!/usr/bin/env bash
set -euo pipefail

allocation_survey_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
allocation_driver_source=$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)

gcc -shared -fPIC -O2 -Wall -Wextra \
  -I"$allocation_driver_source/src/common/sdk/nvidia/inc" \
  -I"$allocation_driver_source/src/common/inc" \
  "$allocation_survey_dir/rm-observe-all.c" -ldl -o /tmp/dgx-rm-observe-all.so
g++ -std=c++17 -O0 -g -Wall -Wextra -I/usr/local/cuda/include \
  "$allocation_survey_dir/probe.cpp" -L/usr/local/cuda/lib64 \
  -Wl,-rpath,/usr/local/cuda/lib64 -lcudart -lcuda \
  -o /tmp/dgx-all-cuda-allocation-probe
python3 "$allocation_survey_dir/run.py"

/usr/local/cuda/bin/nvcc -std=c++17 -arch=sm_121 -O0 -lineinfo \
  "$allocation_survey_dir/device-heap.cu" -o /tmp/dgx-cuda-device-heap-probe
strace -f -yy -tt -T -s 512 \
  -e trace=ioctl,mmap,munmap,mlock,mlock2,munlock,madvise,write \
  -o "$allocation_survey_dir/device-heap.strace" \
  env LD_PRELOAD=/tmp/dgx-rm-observe-all.so /tmp/dgx-cuda-device-heap-probe \
  > "$allocation_survey_dir/device-heap.log" 2>&1
