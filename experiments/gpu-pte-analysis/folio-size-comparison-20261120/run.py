#!/usr/bin/env python3
"""Run equal-size malloc/cudaMalloc tests, optionally with kernel probes."""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--trace', action='store_true', help='requires root/BPF privileges')
ap.add_argument('--sizes', default='1024,4096', help='comma-separated MiB, each <=4096')
ap.add_argument('--repeat', type=int, default=1)
args = ap.parse_args()
sizes = [int(x) for x in args.sizes.split(',')]
if not sizes or any(n <= 0 or n > 4096 for n in sizes) or not 1 <= args.repeat <= 5:
    ap.error('sizes must be 1..4096 MiB and repeat must be 1..5')
if args.trace and os.geteuid() != 0:
    ap.error('--trace requires root; this program never invokes sudo itself')

stamp = datetime.datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')
out = HERE / ('trace' if args.trace else 'baseline') / stamp
out.mkdir(parents=True, exist_ok=False)
binary = HERE / 'folio_compare'
build = subprocess.run([
    'g++', '-std=c++17', '-O2', '-g', '-rdynamic', '-Wall', '-Wextra',
    '-I/usr/local/cuda/include', str(HERE / 'probe.cpp'),
    '-L/usr/local/cuda/lib64', '-Wl,-rpath,/usr/local/cuda/lib64',
    '-lcudart', '-o', str(binary),
], capture_output=True, text=True)
(out / 'build.log').write_text(build.stdout + build.stderr)
if build.returncode:
    raise SystemExit(f'Build failed; see {out / "build.log"}')

env = {'date': stamp, 'kernel': os.uname().release, 'uid': os.getuid(),
       'page_size': os.sysconf('SC_PAGE_SIZE'), 'trace': args.trace,
       'sizes_mib': sizes, 'repeat': args.repeat}
thp = Path('/sys/kernel/mm/transparent_hugepage')
for p in [thp / 'enabled', thp / 'defrag', *sorted(thp.glob('hugepages-*/enabled'))]:
    try:
        env[str(p)] = p.read_text().strip()
    except OSError as e:
        env[str(p)] = str(e)
(out / 'environment.json').write_text(json.dumps(env, indent=2) + '\n')

bt = out / 'trace.bt'
if args.trace:
    bt.write_text((HERE / 'trace.bt.in').read_text().replace('__BINARY__', str(binary)))
def trace_histogram(text, data):
    """Keep ambiguous/internal RM allocations out of the reported histogram."""
    maps = {}
    for line in text.splitlines():
        m = re.fullmatch(r'(@\w+)\[([0-9, ]+)\]: ([0-9]+)', line.strip())
        if m:
            key = tuple(int(x.strip()) for x in m[2].split(','))
            maps.setdefault(m[1], {})[key] = int(m[3])
    hist = {}
    if data['mode'] == 'malloc':
        for (order,), count in maps.get('@anon_folios', {}).items():
            hist[order] = hist.get(order, 0) + count
        lo = int(data['address'], 16) & ~(data['base_page_bytes'] - 1)
        end = int(data['address'], 16) + data['requested_bytes']
        for (address, order), count in maps.get('@anon_alloc_phase', {}).items():
            if lo <= address < end:
                hist[order] = hist.get(order, 0) + count
        method = 'successful anonymous folio allocations within the malloc VA range'
    else:
        requests = maps.get('@rm_requests', {})
        successes = maps.get('@linux_rm_success', {})
        expected = 0
        for (phase, total, block, contig, status), count in requests.items():
            if phase != 1 or total != data['requested_bytes'] or status != 0:
                continue
            # Do not include partial allocations from a failed attempt in the
            # same group; leave the comparison unresolved if that happened.
            if any(p == phase and n == total and b == block and c == contig and s != 0
                   for p, n, b, c, s in requests):
                continue
            expected += count * total
            for (p, n, b, c, leaf, order), allocations in successes.items():
                if (p, n, b, c) == (phase, total, block, contig):
                    hist[order] = hist.get(order, 0) + allocations
        observed = sum((data['base_page_bytes'] << order) * n for order, n in hist.items())
        if expected == 0 or observed != expected:
            return None, 'RM attribution/count cross-check incomplete; inspect trace logs'
        method = ('successful RM Linux allocations matching API phase/size; '
                  'compound-folio interpretation follows nv_compute_gfp_mask()')
    if not hist:
        return None, 'no qualifying allocation events captured'
    rows = [{'order': order, 'folio_bytes': data['base_page_bytes'] << order,
             'successful_allocations': n,
             'allocated_bytes': (data['base_page_bytes'] << order) * n}
            for order, n in sorted(hist.items())]
    return rows, method

results = []
failed = False
for size in sizes:
    for rep in range(args.repeat):
        for mode in ('malloc', 'cudaMalloc'):
            label = f'{mode}-{size}MiB-{rep + 1}'
            child = [str(binary), mode, str(size)]
            cmd = ['bpftrace', str(bt), '-c', shlex.join(child)] if args.trace else child
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            (out / (label + '.stdout.log')).write_text(p.stdout)
            (out / (label + '.stderr.log')).write_text(p.stderr)
            data = None
            for line in p.stdout.splitlines():
                if line.startswith('PROBE_RESULT '):
                    data = json.loads(line.removeprefix('PROBE_RESULT '))
            if data and args.trace and p.returncode == 0:
                hist, method = trace_histogram(p.stdout, data)
                data['physical_folio_histogram'] = hist
                data['physical_folio_method'] = method
                failed |= hist is None
            row = {'label': label, 'returncode': p.returncode, 'observation': data}
            results.append(row)
            failed |= p.returncode != 0 or data is None or not data['data_check_pass']
            if data:
                print(f"{label}: CPU present={data['cpu_pte_present_pages']}/"
                      f"{data['range_base_pages']}; visible PFNs={data['visible_pfn_pages']}; "
                      f"data check={data['data_check_pass']}", flush=True)
            else:
                print(f'{label}: FAILED; inspect logs', flush=True)
(out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
print(f'Artifacts: {out}')
raise SystemExit(1 if failed else 0)
