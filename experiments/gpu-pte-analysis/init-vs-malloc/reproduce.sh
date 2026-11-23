#!/usr/bin/env bash
set -euo pipefail
task_dir=$(cd -- "$(dirname -- "$0")" && pwd)
source_root=$(cd -- "$task_dir/../.." && pwd)
driver_root=$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
g++ -std=c++17 -Wall -Wextra -O2 -I/usr/local/cuda/include "$task_dir/probe.cpp" \
    -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64 -lcudart -lcuda \
    -o /tmp/dgx-init-vs-malloc
gcc -shared -fPIC -Wall -Wextra -O2 "$task_dir/observe.c" -ldl \
    -I"$driver_root/src/common/sdk/nvidia/inc" -I"$driver_root/kernel-open/common/inc" \
    -I"$driver_root/kernel-open/nvidia-uvm" -o /tmp/dgx-init-vs-malloc-observe.so
for task_case in init-only driver staged cold; do
    strace -f -yy -tt -T -s 8192 -e trace=ioctl,mmap,munmap,mlock,munlock,madvise,write \
        -o "$task_dir/$task_case.strace" env LD_PRELOAD=/tmp/dgx-init-vs-malloc-observe.so \
        /tmp/dgx-init-vs-malloc "$task_case" >"$task_dir/$task_case.stdout" 2>"$task_dir/$task_case.log"
done
strace -f -yy -tt -T -s 8192 -e trace=ioctl,mmap,munmap,mlock,mlock2,munlock,madvise,write \
    -o "$task_dir/staged-plain.strace" /tmp/dgx-init-vs-malloc staged \
    > "$task_dir/staged-plain.stdout" 2> "$task_dir/staged-plain.log"
python3 "$task_dir/analyze.py"
