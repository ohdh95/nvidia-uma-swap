#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
run_id=$(TZ=Asia/Seoul date +%Y%m%dT%H%M%S)-$$
log_dir="$PWD/results-$run_id"
mkdir -- "$log_dir"
bash build.sh
{
    TZ=Asia/Seoul date --iso-8601=seconds
    uname -a
    /usr/local/cuda/bin/nvcc --version
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    nvidia-smi -q | rg 'Addressing Mode'
    getconf PAGESIZE
    cat /proc/swaps
    cat /sys/module/zswap/parameters/enabled
} > "$log_dir/environment.txt"

./malloc_pageout resident 2>&1 | tee "$log_dir/resident.log"
./malloc_pageout pageout 2>&1 | tee "$log_dir/pageout.log"
./malloc_pageout pageout warm 2>&1 | tee "$log_dir/pageout-warm.log"

run_reclaim() {
    local label=$1
    shift
    systemd-run --user --wait --pipe --collect \
        --unit="malloc-pageout-gpu-$run_id-$label" \
        --property=Delegate=yes --working-directory="$PWD" \
        "$PWD/malloc_pageout" pageout reclaim "$@" \
        2>&1 | tee "$log_dir/$label.log"
}
run_reclaim pageout-reclaim
run_reclaim pageout-reclaim-warm warm
printf 'Logs: %s\n' "$log_dir"
