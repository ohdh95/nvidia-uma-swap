#!/usr/bin/env bash
set -euo pipefail
artifact_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cc -O2 -Wall -Wextra "${artifact_dir}/swap-pte-check.c" -o /tmp/dgx-swap-pte-check
/tmp/dgx-swap-pte-check 2>&1 | tee "${artifact_dir}/run.log"
