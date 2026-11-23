cudaMalloc vs malloc: physical folio size comparison on DGX Spark

Scope
  cudaMalloc is compared with ordinary malloc, not cudaMallocManaged.
  The test does not change kernel/driver code, THP settings, LRU membership,
  swap configuration, or module state.

Programs
  probe.cpp: allocates one requested range in a fresh process. malloc is
    materialized by CPU memset and one checked byte per base page. cudaMalloc
    is materialized by cudaMemset + synchronization, then checked at both ends
    with device-to-host copies. The device pointer is never CPU-dereferenced.
  run.py: compiles probe.cpp, runs equal sizes, and saves observations/logs.
  trace.bt.in: optional read-only kernel probes, filtered to the child PID and
    explicitly marked API/touch phases. CUDA initialization/warmup is excluded.

Commands
  Baseline, no privileged tracing:
    python3 experiments/gpu-pte-analysis/folio-size-comparison-20261120/run.py

  Physical allocation-order tracing, requires root:
    sudo python3 experiments/gpu-pte-analysis/folio-size-comparison-20261120/run.py --trace --sizes 1024

  To include 4 GiB or repeat runs:
    add --sizes 1024,4096 --repeat 3

Measurement meaning
  - Ordinary malloc: successful vma_alloc_folio_noprof returns for faults in
    the requested VA range identify anonymous folio allocation orders.
    Allocation-phase fault addresses are filtered after malloc returns.
  - cudaMalloc: its returned VA has no CPU PTE on this machine, including after
    GPU writes. Neither CPU smaps page-size fields nor pagemap PFNs reveal
    its backing. The tracer instead records RM nv_alloc_pages requests/status
    and successful Linux alloc_pages_noprof/__alloc_pages_noprof calls inside
    nv_alloc_system_pages/nv_alloc_contig_pages. Both wrappers are required
    because CONFIG_NUMA's alloc_pages() does not call __alloc_pages_noprof().
  - The CUDA histogram is emitted only when a successful allocation-phase RM
    request matches the API size and corresponding Linux allocation bytes
    cross-check. Unresolved cases retain null and need inspection of raw logs;
    internal allocations must not be silently counted as application backing.
  - The local NVIDIA nv_compute_gfp_mask() adds __GFP_COMP for order > 0.
    Therefore those successful RM allocation blocks correspond to compound
    folios. Anonymous vma_alloc_folio calls likewise request folios.
    folio_bytes = base_page_bytes * 2**order.
  - This measures allocation-time granularity, not a later snapshot after
    folio migration/splitting. Partial allocations from failed RM attempts are
    not interpreted as a successful application allocation.
  - smaps metrics describe entire intersecting VMAs, which may include other
    mappings or the inspection buffers. KernelPageSize/MMUPageSize are not
    used as a physical folio-size measurement.
  - The rounded CPU range can contain one extra base page because malloc's
    returned pointer is not necessarily page aligned. Requested sizes are
    nevertheless exactly equal between the two APIs.
