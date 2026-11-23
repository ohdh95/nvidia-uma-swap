#!/usr/bin/env python3
"""Convert the installed Nsight Compute GB20B catalog to searchable lists."""
import collections
import csv
import gzip
import json
import pathlib
import re
import shutil

root = pathlib.Path(__file__).resolve().parent
row_pattern = re.compile(r"^(\S+__\S+)\s+(Counter|Ratio|Throughput)\s+")

def catalog_rows(path):
    rows = []
    for line in path.read_text().splitlines():
        match = row_pattern.match(line)
        if match:
            offset = match.start(2)
            rows.append({
                "metric": match[1], "type": match[2],
                "unit": line[offset + 16:offset + 32].strip(),
                "description": line[offset + 32:].strip(),
            })
        elif line[:1].isspace() and rows:
            continuation = line.strip()
            if continuation:
                rows[-1]["description"] += " " + continuation
    return rows

def write_tsv(name, rows, fields):
    with (root / name).open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

base = catalog_rows(root / "ncu-gb20b-base.txt")
by_name = {row["metric"]: row for row in base}
assert len(base) == len(by_name) and base
write_tsv("gb20b-base-metrics.tsv", base, ["metric", "type", "unit", "description"])
memory = [row for row in base if row["metric"].startswith(("l1tex__", "lts__", "gcc__", "idc__"))
          or any(term in row["metric"] for term in ("memory", "scoreboard"))]
write_tsv("gb20b-memory-metrics.tsv", memory, ["metric", "type", "unit", "description"])

choices = [
    ("실행 시간", "gpu__time_duration.sum", "커널 실행 시간"),
    ("Global read", "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", "global load가 요청한 sector 수"),
    ("Global write", "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum", "global store가 요청한 sector 수"),
    ("L1/TEX hit", "l1tex__t_sector_hit_rate.pct", "L1/TEX sector 적중률"),
    ("L2 read", "lts__t_sectors_op_read.sum", "L2 read sector 수"),
    ("L2 write", "lts__t_sectors_op_write.sum", "L2 write sector 수"),
    ("Sysmem read", "lts__t_sectors_aperture_sysmem_op_read.sum", "sysmem aperture에 대한 L2 read sector 수"),
    ("Sysmem write", "lts__t_sectors_aperture_sysmem_op_write.sum", "sysmem aperture에 대한 L2 write sector 수"),
    ("L2 hit", "lts__t_sector_hit_rate.pct", "L2 sector 적중률"),
    ("Memory dependency stall", "smsp__warps_issue_stalled_long_scoreboard.sum", "L1TEX 의존성으로 대기한 warp 누적 수"),
    ("Occupancy", "sm__warps_active.avg.pct_of_peak_sustained_active", "active 시간 기준 warp occupancy"),
    ("Instructions", "smsp__inst_executed.sum", "실행한 warp 명령어 수"),
    ("SM activity", "sm__cycles_active.avg.pct_of_peak_sustained_elapsed", "SM이 active였던 cycle 비율"),
]

# The all-mode output contains suffix/rollup variants, not independent counters.
raw_all = root / "ncu-gb20b-all.txt"
compressed_all = root / "ncu-gb20b-all.txt.gz"
all_names = set()
input_stream = raw_all.open() if raw_all.exists() else gzip.open(compressed_all, "rt")
with input_stream as source, (root / "gb20b-full-metric-names.txt").open("w") as output:
    for line in source:
        match = row_pattern.match(line)
        if match:
            assert match[1] not in all_names
            all_names.add(match[1])
            output.write(match[1] + "\n")
selected = []
for category, name, explanation in choices:
    assert name in all_names, name
    base_name = name.split(".", 1)[0]
    selected.append({"category": category, "metric": name, "meaning": explanation,
                     "base_description": by_name[base_name]["description"]})
write_tsv("selected-metrics.tsv", selected, ["category", "metric", "meaning", "base_description"])

stats = {
    "catalog_source": "Nsight Compute 2025.3.1 offline chip catalog (--chips gb20b)",
    "live_collection_status": "permission denied: ERR_NVGPUCTRPERM; RmProfilingAdminOnly=1",
    "base_metrics": len(base), "full_metric_variants": len(all_names),
    "metric_types": dict(collections.Counter(row["type"] for row in base)),
    "units": dict(collections.Counter(row["metric"].split("__")[0] for row in base)),
    "memory_metric_subset": len(memory),
    "has_dram_prefix": any(row["metric"].startswith("dram__") for row in base),
    "translation_names": [row["metric"] for row in base if re.search(r"(?:^|_)(?:tlb|mmu|gmmu|utlb|fault|pte|page)(?:_|$)", row["metric"])],
    "pmsampling_base_metrics": len(catalog_rows(root / "ncu-gb20b-pmsampling.txt")),
    "warpsampling_base_metrics": len(catalog_rows(root / "ncu-gb20b-warpsampling.txt")),
}
(root / "catalog-summary.json").write_text(json.dumps(stats, indent=2) + "\n")
fields = re.findall(r'^"([^"\n]+)"', (root / "nvidia-smi-fields.txt").read_text(), re.MULTILINE)
(root / "nvidia-smi-field-names.txt").write_text("\n".join(fields) + "\n")

if raw_all.exists():
    with raw_all.open("rb") as source, gzip.open(compressed_all, "wb") as target:
        shutil.copyfileobj(source, target)
    raw_all.unlink()  # Only this script's generated 93-MB raw query output.
print(json.dumps(stats, indent=2))
