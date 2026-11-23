#!/usr/bin/env bash
set -euo pipefail
task_dir=$(cd -- "$(dirname -- "$0")" && pwd)
source_root=$(cd -- "$task_dir/../.." && pwd)
driver_root=$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
gcc -shared -fPIC -O2 -Wall -Wextra "$task_dir/query.c" -ldl \
    -I"$driver_root/src/common/sdk/nvidia/inc" -I"$driver_root/kernel-open/common/inc" \
    -o /tmp/dgx-access-counter-query.so
g++ -O2 -std=c++17 -I/usr/local/cuda/include "$task_dir/../init-vs-malloc/probe.cpp" \
    -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64 -lcudart -lcuda \
    -o /tmp/dgx-access-counter-probe
strace -f -yy -tt -T -s 1024 -e trace=ioctl,write -o "$task_dir/query.strace" \
    env LD_PRELOAD=/tmp/dgx-access-counter-query.so /tmp/dgx-access-counter-probe init-only \
    > "$task_dir/query.stdout" 2> "$task_dir/query.log"
cat "$task_dir/query.log"
