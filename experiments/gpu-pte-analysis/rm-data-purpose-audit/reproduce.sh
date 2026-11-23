#!/usr/bin/env bash
set -euo pipefail
purpose_dir=$(cd -- "$(dirname -- "$0")" && pwd)
purpose_source_root=$(cd -- "$purpose_dir/../.." && pwd)
purpose_driver=$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
g++ -std=c++17 -Wall -Wextra -O2 -I/usr/local/cuda/include \
    "$purpose_dir/../init-vs-malloc/probe.cpp" -L/usr/local/cuda/lib64 \
    -Wl,-rpath,/usr/local/cuda/lib64 -lcudart -lcuda -o /tmp/dgx-rm-purpose-workload
gcc -shared -fPIC -O2 -Wall -Wextra "$purpose_dir/observe.c" -ldl \
    -I"$purpose_driver/src/common/sdk/nvidia/inc" \
    -I"$purpose_driver/kernel-open/common/inc" -I"$purpose_driver/kernel-open/nvidia-uvm" \
    -o /tmp/dgx-rm-data-purpose-observer.so
for purpose_mode in staged cold; do
    strace -f -yy -tt -T -s 1024 -e trace=ioctl,mmap,munmap,mlock,munlock,write \
        -o "$purpose_dir/$purpose_mode.strace" \
        env LD_PRELOAD=/tmp/dgx-rm-data-purpose-observer.so /tmp/dgx-rm-purpose-workload "$purpose_mode" \
        >"$purpose_dir/$purpose_mode.stdout" 2>"$purpose_dir/$purpose_mode.log"
done
python3 "$purpose_dir/analyze.py"
