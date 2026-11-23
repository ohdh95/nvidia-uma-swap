#!/usr/bin/env python3
"""Check audit source locations and record current read-only environment facts."""
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[4]
DRIVER = ROOT / 'nvidia-uma-swap'
KERNEL = ROOT / 'linux-uma-swap'
errors = []
rows = list(csv.DictReader((OUT / 'allocation-sites.tsv').open(), delimiter='\t'))
for row in rows:
    path = ROOT / row['file']
    lines = path.read_text(errors='replace').splitlines()
    line = int(row['line'])
    if not 1 <= line <= len(lines) or row['symbol'] not in lines[line - 1]:
        errors.append({'index_location': row})

report = (OUT / 'report.ko.txt').read_text()
refs = re.findall(r'\b((?:kernel-open|src|kernel)/[\w./-]+\.[ch]):([0-9, -]+)', report)
report_sources = {}
reference_count = 0
for relative, numbers in refs:
    path = KERNEL / relative[7:] if relative.startswith('kernel/') else DRIVER / relative
    if not path.is_file():
        errors.append({'missing_report_source': str(path)})
        continue
    data = path.read_bytes()
    lines = data.decode(errors='replace').splitlines()
    for number in re.findall(r'\d+', numbers):
        reference_count += 1
        if not 1 <= int(number) <= len(lines):
            errors.append({'invalid_report_line': str(path), 'line': number})
    report_sources[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
with (OUT / 'report-source-sha256.tsv').open('w', newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(['file', 'sha256'])
    writer.writerows(sorted(report_sources.items()))

def command(args):
    result = subprocess.run(args, capture_output=True, text=True)
    return {'returncode': result.returncode, 'stdout': result.stdout.strip(),
            'stderr': result.stderr.strip()}

facts = {
    'observation_date_Asia_Seoul': '2026-11-20',
    'kernel': command(['uname', '-r']),
    'page_size': command(['getconf', 'PAGESIZE']),
    'gpu_addressing': command(['nvidia-smi', '--query-gpu=name,addressing_mode', '--format=csv,noheader']),
}
paths = [
    '/proc/driver/nvidia/version', '/sys/devices/system/node/online',
    '/sys/bus/pci/devices/000f:01:00.0/numa_node',
    '/sys/kernel/iommu_groups/20/type', '/sys/kernel/mm/lru_gen/enabled',
    '/sys/kernel/mm/transparent_hugepage/enabled', '/proc/swaps',
]
paths += ['/sys/module/nvidia_uvm/parameters/' + p for p in [
    'uvm_ats_mode', 'uvm_disable_hmm', 'uvm_cpu_chunk_allocation_sizes',
    'uvm_page_table_location', 'uvm_enable_va_space_mm', 'uvm_global_oversubscription',
]]
facts['read_only_values'] = {p: Path(p).read_text().strip() for p in paths}
(OUT / 'environment.json').write_text(json.dumps(facts, ensure_ascii=False, indent=2) + '\n')
validation = {
    'index_source_locations_checked': len(rows),
    'explicit_report_source_files_checked': len(report_sources),
    'explicit_report_source_line_references_checked': reference_count,
    'errors': errors,
    'limits': 'Location/hash checks only. Semantic paths manually reviewed; no runtime frame/folio-flag tracing.',
}
(OUT / 'validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(validation, ensure_ascii=False, indent=2))
raise SystemExit(bool(errors))
