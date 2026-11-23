import collections
import csv
import json
import re
import sys
from pathlib import Path

run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
text = (run_dir / 'run.log').read_text()
rows = list(csv.DictReader((run_dir / 'uvm-events.csv').open()))
allocation = re.search(r'ALLOCATION base=(\S+) end=(\S+) bytes=(\d+) pages=(\d+)', text)
if allocation is None:
    raise SystemExit('Missing allocation record')
summary = {
    'allocation': dict(zip(('base', 'end', 'bytes', 'base_pages'), allocation.groups())),
    'event_source': 'Per-process UVM tools event tracker V2',
    'phases': {},
}
for match in re.finditer(r'EVENTS phase=(\S+) gpu_faults=(\d+) target_gpu_faults=(\d+) '
                        r'unique_target_4k_pages=(\d+) cpu_faults=(\d+) replays=(\d+) dropped_total=(\d+)', text):
    phase = match[1]
    record = dict(zip(('gpu_faults', 'target_gpu_faults', 'unique_target_4k_pages',
                      'cpu_faults', 'replays', 'dropped_total'), map(int, match.groups()[1:])))
    faults = [r for r in rows if r['phase'] == phase and r['type'] == 'gpu_fault']
    record['gpu_fault_types'] = dict(collections.Counter(r['fault_type'] for r in faults))
    record['gpu_access_types'] = dict(collections.Counter(r['access_type'] for r in faults))
    record['gpu_faults_outside_target'] = sum(r['in_allocation'] != '1' for r in faults)
    summary['phases'][phase] = record
for match in re.finditer(r'CPU_PTES phase=(\S+) present=(\d+)/(\d+) swapped=(\d+)', text):
    summary['phases'][match[1]]['cpu_ptes'] = dict(zip(
        ('present', 'total', 'swapped'), map(int, match.groups()[1:])))
verify = re.search(r'VERIFY pages=(\d+) mismatches=(\d+)', text)
summary['verified_values'] = int(verify[1]) if verify else None
summary['mismatches'] = int(verify[2]) if verify else None
phases = summary['phases']
checks = {
    'program_pass': 'RESULT PASS' in text,
    'no_initial_cpu_mapping': phases['after_alloc_before_access']['cpu_ptes']['present'] == 0,
    'no_initial_target_gpu_faults': phases['after_alloc_before_access']['target_gpu_faults'] == 0,
    'first_gpu_access_faulted': phases['gpu_first_touch']['target_gpu_faults'] > 0,
    'first_faults_have_pde_or_pte_cause': bool(phases['gpu_first_touch']['gpu_fault_types']) and
        set(phases['gpu_first_touch']['gpu_fault_types']) <= {'1', '2'},
    'second_gpu_access_did_not_fault': phases['gpu_second_touch']['target_gpu_faults'] == 0,
    'no_cpu_faults_during_gpu_accesses': phases['gpu_first_touch']['cpu_faults'] == 0 and
        phases['gpu_second_touch']['cpu_faults'] == 0,
    'all_gpu_faults_in_target': all(r['gpu_faults_outside_target'] == 0 for r in phases.values()),
    'no_dropped_events': all(r['dropped_total'] == 0 for r in phases.values()),
    'data_correct': summary['mismatches'] == 0 and summary['verified_values'] == 4096,
}
summary['fault_type_names'] = {'1': 'UvmFaultTypeInvalidPde', '2': 'UvmFaultTypeInvalidPte'}
summary['checks'] = checks
summary['pass'] = all(checks.values())
(run_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print('PASS' if summary['pass'] else 'FAIL', json.dumps(checks))
raise SystemExit(0 if summary['pass'] else 1)
