#!/usr/bin/env python3
"""Summarize observed API phases without assuming one ioctl per allocation."""
import collections
import json
import pathlib
import re

root = pathlib.Path(__file__).resolve().parent
modes = ("init-only", "driver", "staged", "cold")
observations = {}
summary = ["mode\tphase\trequested_bytes\trm_sysmem_count\trm_input_bytes\trm_output_bytes\tuvm_map_count\trm_object_count\tioctl_count"]

def trace_sections(path):
    sections = collections.defaultdict(list)
    phase = "outside"
    for line in path.read_text().splitlines():
        match = re.search(r"PHASE_BEGIN ([A-Za-z0-9_]+) requested_bytes=", line)
        if match:
            phase = match[1]
        sections[phase].append(line)
        if "PHASE_END " in line:
            phase = "outside"
    return sections

for mode in modes:
    phases = collections.OrderedDict()
    phase = "outside"
    stacks = {}
    stack_object = None
    lines = (root / f"{mode}.log").read_text().splitlines()
    assert f"RESULT PASS {mode}" in lines, mode
    for line_number, line in enumerate(lines, 1):
        if line.startswith("PHASE_BEGIN "):
            match = re.fullmatch(r"PHASE_BEGIN (\S+) requested_bytes=(\d+)", line)
            phase = match[1]
            phases[phase] = {"requested_bytes": int(match[2]), "events": [], "returns": [], "line": line_number}
        elif line.startswith("PHASE_END "):
            phase = "outside"
        elif line.startswith("OBS "):
            event = json.loads(line[4:])
            event["line"] = line_number
            phases.setdefault(phase, {"requested_bytes": 0, "events": [], "returns": []})["events"].append(event)
        elif line.startswith("RETURN "):
            phases[phase]["returns"].append(line)
        elif line.startswith("STACK_BEGIN "):
            stack_object = int(line.split("object=")[1], 16)
            stacks[stack_object] = []
        elif line == "STACK_END":
            stack_object = None
        elif stack_object is not None:
            stacks[stack_object].append(line)
    sections = trace_sections(root / f"{mode}.strace")
    for name, entry in phases.items():
        successful = [e for e in entry["events"] if e["result"] == 0 and e["status"] == 0]
        memory = [e for e in successful if e["event"] == "rm_memory"]
        entry.update({
            "rm_sysmem_count": len(memory),
            "rm_input_bytes": sum(e["in"]["size"] for e in memory),
            "rm_output_bytes": sum(e["out"]["size"] for e in memory),
            "uvm_map_count": sum(e["event"] == "uvm_map_external" for e in successful),
            "rm_object_count": sum(e["event"] == "rm_object" for e in successful),
            "ioctl_count": sum("ioctl(" in line for line in sections[name]),
        })
        summary.append("\t".join(str(x) for x in (
            mode, name, entry["requested_bytes"], entry["rm_sysmem_count"], entry["rm_input_bytes"],
            entry["rm_output_bytes"], entry["uvm_map_count"], entry["rm_object_count"], entry["ioctl_count"])))
        if name in ("cuInit", "cudaSetDevice", "cudaMalloc_0", "cudaMalloc_1", "cudaMalloc_2", "cudaMalloc_3", "cudaMalloc_4", "cudaMalloc_5"):
            (root / f"{mode}.{name}.strace").write_text("\n".join(sections[name]) + "\n")
    observations[mode] = {"phases": phases, "stacks": stacks}

staged = observations["staged"]["phases"]
target = next(e for e in staged["cudaMalloc_3"]["events"] if e["event"] == "rm_memory")
initial = [e for e in staged["cudaSetDevice"]["events"] if e["event"] == "rm_memory"]
same_input = [e for e in initial if e["in"] == target["in"]]
same_output = [e for e in same_input if e["out"] == target["out"]]
assert same_input and same_output
assert all(all(e[k] == target[k] for k in ("client", "parent", "class", "paramsSize", "rights")) for e in same_input)
map_fields = ("length", "offset", "gpuCount", "mappingType", "cachingType", "formatType", "elementBits", "compressionType")
all_maps = [e for v in staged.values() for e in v["events"] if e["event"] == "uvm_map_external"]
target_map = next(e for e in all_maps if e["object"] == target["object"])
same_maps = [e for e in all_maps if any(m["object"] == e["object"] for m in same_input)
             and all(e[k] == target_map[k] for k in map_fields)]
comparison = {
    "target": target,
    "initial_allocations_with_identical_input": same_input,
    "initial_allocations_with_identical_output": same_output,
    "initial_mappings_with_identical_properties": same_maps,
    "initial_type_counts": dict(collections.Counter(e["in"]["type"] for e in initial)),
    "same_header_fields": ["client", "parent", "class", "paramsSize", "rights"],
    "different_identifiers": "new object handle, user argument pointer and mapped VA are allocation identities, not purpose tags",
}

# The original process also runs without the decoder: compare driver syscall
# counts in each phase, excluding the decoder's own writes/backtrace syscalls.
plain = trace_sections(root / "staged-plain.strace")
instrumented = trace_sections(root / "staged.strace")
plain_counts = {name: sum("ioctl(" in line for line in rows) for name, rows in plain.items() if name != "outside"}
decoder_counts = {name: sum("ioctl(" in line for line in rows) for name, rows in instrumented.items() if name != "outside"}
assert plain_counts == decoder_counts, (plain_counts, decoder_counts)
comparison["plain_and_decoder_ioctl_counts_equal"] = plain_counts

(root / "summary.tsv").write_text("\n".join(summary) + "\n")
(root / "observations.json").write_text(json.dumps(observations, indent=2) + "\n")
(root / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
print("All four probes passed; plain and decoder ioctl counts match.")
print("Identical initialization/data allocation inputs:", len(same_input))
print("Identical initialization/data allocation outputs:", len(same_output))
print("Identical initialization/data mapping properties:", len(same_maps))
print("\n".join(summary))
