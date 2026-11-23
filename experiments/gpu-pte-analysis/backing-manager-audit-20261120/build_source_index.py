#!/usr/bin/env python3
"""Reproducible lexical allocation inventory; not a compiled call graph."""
import bisect
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
DRIVER = ROOT / 'nvidia-uma-swap'
KERNEL = ROOT / 'linux-uma-swap'

# Erase comments and string/character literals while preserving offsets/lines.
ERASE = re.compile(r'/\*[\s\S]*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')

def code_only(src):
    return ERASE.sub(lambda m: ''.join('\n' if c == '\n' else ' ' for c in m.group()), src)

PRIMITIVES = {
    'alloc_page', 'alloc_pages', 'alloc_pages_node', 'alloc_pages_exact',
    '__alloc_pages', '__get_free_page', '__get_free_pages', 'get_zeroed_page',
    'dma_alloc_coherent', 'dma_alloc_attrs', 'dma_alloc_noncontiguous',
    'kmalloc', 'kzalloc', 'kcalloc', 'kmalloc_array', 'krealloc',
    'vmalloc', 'vzalloc', 'kvmalloc', 'kvzalloc', 'kvcalloc', 'kvmalloc_array',
    'kmem_cache_alloc', 'kmem_cache_zalloc', 'nv_kmem_cache_alloc',
    'nv_kmem_cache_zalloc', 'NV_KMALLOC', 'NV_KZALLOC', 'NV_KMALLOC_ATOMIC',
    'NV_KMALLOC_NO_OOM', 'NV_ALLOC_PAGES_NODE', 'NV_GET_FREE_PAGES', 'nv_vmalloc',
}
BACKING = {
    'memdescAlloc', 'memdescAllocList', 'memdescTagAlloc', 'memdescTagAllocList',
    'osAllocPages', 'osAllocPagesInternal', 'osAllocPagesNode', 'os_alloc_pages_node',
    'nv_alloc_pages', 'nv_alloc_system_pages', 'nv_alloc_contig_pages',
    'nv_alloc_coherent_pages', 'uvm_cpu_chunk_alloc', 'uvm_cpu_chunk_alloc_page',
    'uvm_cpu_chunk_alloc_hmm', 'uvm_cpu_chunk_split', 'uvm_cpu_chunk_merge',
    'uvm_mem_alloc', 'uvm_mem_alloc_sysmem', 'uvm_mem_alloc_sysmem_dma',
    'uvm_mem_alloc_vidmem', 'uvm_mem_alloc_sysmem_and_map_cpu_kernel',
    'uvm_mem_alloc_sysmem_dma_and_map_cpu_kernel',
    'uvm_rm_mem_alloc', 'uvm_rm_mem_alloc_and_map_cpu', 'uvm_rm_mem_alloc_and_map_all',
    'uvm_pmm_gpu_alloc_user', 'uvm_pmm_gpu_alloc_kernel', 'uvm_gpu_dma_alloc',
    'nvUvmInterfaceMemoryAllocSys', 'nvUvmInterfaceMemoryAllocFB',
    'nvUvmInterfacePmaAllocPages', 'pmaAllocatePages', 'ctxBufPoolAllocate',
    'heapAlloc', 'memdescCreateSubMem', 'memdescDescribe',
    'os_alloc_mem', 'portMemAllocNonPaged', 'portMemAllocPaged',
}
EXISTING = {
    'NV_PIN_USER_PAGES', 'NV_PIN_USER_PAGES_REMOTE', 'NV_GET_USER_PAGES_REMOTE',
    'nv_alias_pages', 'nv_register_user_pages', 'nv_register_phys_pages',
    'nv_register_peer_io_mem', 'nv_register_sgt', 'os_lock_user_pages',
    'nv_dma_import_from_fd', 'nv_dma_import_dma_buf', 'nv_drm_lock_user_pages',
    'uvm_populate_pageable_vma', 'uvm_migrate_pageable', 'migrate_vma_setup',
    'migrate_vma_pages', 'migrate_vma_finalize',
}
ALL = PRIMITIVES | BACKING | EXISTING
PATTERN = re.compile(r'\b(' + '|'.join(sorted(ALL, key=len, reverse=True)) + r')\s*\(')

def source_context(path):
    s = str(path.relative_to(DRIVER))
    if 'test' in path.name or 'selftest' in path.name:
        return 'test_source_or_helper; verify caller gates'
    if path.name == 'uvm_hmm.c':
        return 'HMM_disabled_on_current_system'
    if path.name == 'uvm_ats_sva.c':
        return 'SMMU_WAR_allocation_conditional_on_kernel_conftest'
    if path.name == 'linux_nvswitch.c':
        return 'NVSwitch_not_present_on_current_single_GB10_system'
    if '/phys_mem_allocator/numa.c' in s:
        return 'PMA_GPU_NUMA_memory_not_current_Spark_configuration'
    if '/vgpu/' in s or s.startswith('kernel-open/nvidia-vgpu'):
        return 'virtualization_path_not_current_baremetal_configuration'
    return 'source_candidate; activity_requires_branch_or_runtime_evidence'

files = set((DRIVER / 'kernel-open').rglob('*.c'))
files.update((DRIVER / 'src/nvidia/src/kernel').rglob('*.c'))
files.update((DRIVER / 'src/nvidia/arch/nvalloc/unix/src').glob('*.c'))
files.add(DRIVER / 'src/nvidia/src/libraries/nvport/memory/memory_unix_kernel_os.c')
files.add(DRIVER / 'src/nvidia-modeset/kapi/src/nvkms-kapi.c')
files.add(DRIVER / 'src/nvidia-modeset/src/nvkms-rm.c')
rows = []
hashes = []
scanned = 0
for path in sorted(files):
    if any(p in {'internal', 'library'} for p in path.relative_to(DRIVER).parts):
        continue
    src = path.read_text(errors='replace')
    code = code_only(src)
    starts = [0] + [m.end() for m in re.finditer('\n', src)]
    lines = src.splitlines()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    scanned += 1
    found = False
    for m in PATTERN.finditer(code):
        symbol = m.group(1)
        line = bisect.bisect_right(starts, m.start())
        # A lexical index intentionally includes definitions. They are labeled
        # where the token after the balanced argument list is an opening brace.
        depth, end = 1, m.end()
        while depth and end < len(code):
            if code[end] == '(':
                depth += 1
            elif code[end] == ')':
                depth -= 1
            end += 1
        tail = code[end:].lstrip()
        kind = 'definition' if tail.startswith('{') else 'reference'
        category = ('primitive_or_heap' if symbol in PRIMITIVES else
                    'existing_pages_or_linux_handoff' if symbol in EXISTING else
                    'backing_entry_or_suballocator')
        rows.append([str(path.relative_to(ROOT)), line, symbol, kind, category,
                     source_context(path), lines[line - 1].strip()])
        found = True
    if found:
        hashes.append([str(path.relative_to(ROOT)), digest])

with (OUT / 'allocation-sites.tsv').open('w', newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(['file', 'line', 'symbol', 'kind', 'category', 'context', 'source_line'])
    writer.writerows(rows)
with (OUT / 'source-sha256.tsv').open('w', newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(['file', 'sha256'])
    writer.writerows(hashes)

summary = {
    'scope': 'Public driver source allocation references. Comments/literals erased. No preprocessor evaluation or call-graph proof.',
    'driver_c_files_scanned': scanned,
    'files_with_indexed_references': len(hashes),
    'total_index_rows': len(rows),
    'references_by_category': dict(Counter(r[4] for r in rows if r[3] == 'reference')),
    'primitive_symbols': sorted(PRIMITIVES),
    'definitions_included_and_labeled': True,
    'test_and_inactive_source_kept_with_context': True,
}
(OUT / 'index-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({k:v for k,v in summary.items() if k != 'primitive_symbols'}, indent=2))
