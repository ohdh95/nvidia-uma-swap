"""Correlate observed RM roles without using CUDA API return addresses."""
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def analyze(mode):
    events = []
    phase = None
    lines = (ROOT / f"{mode}.log").read_text().splitlines()
    for line in lines:
        if line.startswith("PHASE_BEGIN "):
            phase = line.split()[1]
        if line.startswith("OBS "):
            event = json.loads(line[4:])
            event["phase"] = phase  # Workload ground truth, not a role criterion.
            events.append(event)
        if line.startswith("PHASE_END "):
            phase = None

    allocations = {
        (e["client"], e["object"]): e
        for e in events
        if e["event"] in ("rm_memory", "rm_alloc_legacy")
        and e["result"] == 0 and e["status"] == 0
    }
    mappings = [e for e in events if e["event"] == "uvm_map_external"
                and e["result"] == 0 and e["status"] == 0]
    roles = []
    for channel in (e for e in events if e["event"] == "rm_channel" and e["result"] == 0):
        for role, field in (("USERD", "userd_handle"), ("ERROR_NOTIFIER", "error_handle")):
            key = (channel["client"], channel[field])
            if not channel[field]:
                continue
            allocation = allocations.get(key)
            roles.append({
                "role": role, "client": key[0], "memory_object": key[1],
                "channel": channel["object"],
                "offset": channel["userd_offset"] if role == "USERD" else 0,
                "allocation_kind": allocation["event"] if allocation else None,
                # Do not infer a role length from allocation length or offset stride.
                "allocation_bytes": (allocation["out"]["size"] if allocation and
                    allocation["event"] == "rm_memory" else None),
            })
        va = channel["gpfifo_va"]
        for mapping in mappings:
            if mapping["client"] != channel["client"]:
                continue
            if not mapping["base"] <= va < mapping["base"] + mapping["length"]:
                continue
            key = (mapping["client"], mapping["object"])
            allocation = allocations.get(key)
            roles.append({
                "role": "GPFIFO", "client": key[0], "memory_object": key[1],
                "channel": channel["object"],
                "offset": mapping["offset"] + va - mapping["base"],
                "entries": channel["gpfifo_entries"],
                "allocation_kind": allocation["event"] if allocation else None,
                "allocation_class": allocation["class"] if allocation else None,
                "allocation_bytes": (allocation["limit"] + 1 if allocation and
                    allocation["event"] == "rm_alloc_legacy" else None),
            })

    userd_keys = {(r["client"], r["memory_object"]) for r in roles if r["role"] == "USERD"}
    cpu_userd = [e for e in events if e["event"] == "rm_map_cpu" and
                (e["client"], e["object"]) in userd_keys and e["result"] == 0 and e["status"] == 0]
    summary = {
        "workload_pass": f"RESULT PASS {mode}" in lines,
        "event_counts": dict(collections.Counter(e["event"] for e in events)),
        "unique_role_backings": {role: len({(r["client"], r["memory_object"])
            for r in roles if r["role"] == role}) for role in ("USERD", "ERROR_NOTIFIER", "GPFIFO")},
        "userd_cpu_mapping_requests": cpu_userd,
        "control_command_counts": {hex(k): v for k, v in collections.Counter(
            e["cmd"] for e in events if e["event"] == "rm_control").items()},
    }
    if mode == "staged":
        init = [e for e in events if e["event"] == "rm_memory" and e["phase"] == "cudaSetDevice"]
        user = next(e for e in events if e["event"] == "rm_memory" and e["phase"] == "cudaMalloc_3")
        same = [e for e in init if e["in"] == user["in"] and e["out"] == user["out"]]
        known = {(r["client"], r["memory_object"]) for r in roles
                 if r["role"] in ("USERD", "ERROR_NOTIFIER")}
        summary.update({
            "init_system_allocation_count": len(init),
            "init_system_objects_with_confirmed_control_roles": sum(
                (e["client"], e["object"]) in known for e in init),
            "init_system_objects_without_confirmed_role": sum(
                (e["client"], e["object"]) not in known for e in init),
            "init_objects_with_identical_memory_params_to_user_2MiB": [hex(e["object"]) for e in same],
            "user_2MiB_object_for_comparison": hex(user["object"]),
        })
    (ROOT / f"{mode}-events.json").write_text(json.dumps(events, indent=2) + "\n")
    (ROOT / f"{mode}-roles.json").write_text(json.dumps(roles, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    summary = {mode: analyze(mode) for mode in ("staged", "cold")}
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for mode, result in summary.items():
        print(mode, "PASS" if result["workload_pass"] else "FAIL", result["unique_role_backings"])
    print("staged init role-known:", summary["staged"]["init_system_objects_with_confirmed_control_roles"])
    print("staged init role-unknown:", summary["staged"]["init_system_objects_without_confirmed_role"])
