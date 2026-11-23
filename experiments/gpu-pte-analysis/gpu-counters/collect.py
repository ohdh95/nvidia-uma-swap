#!/usr/bin/env python3
"""Read-only GPU counter catalog and telemetry discovery; no settings changed."""
import json
import pathlib
import subprocess
import sys

root = pathlib.Path(__file__).resolve().parent
queries = [
    ("ncu-version.txt", ["ncu", "--version"]),
    ("nsys-version.txt", ["nsys", "--version"]),
    ("ncu-base.txt", ["ncu", "--query-metrics", "--query-metrics-mode", "base"]),
    ("ncu-all.txt", ["ncu", "--query-metrics", "--query-metrics-mode", "all"]),
    ("ncu-gb20b-base.txt", ["ncu", "--query-metrics", "--chips", "gb20b", "--query-metrics-mode", "base"]),
    ("ncu-gb20b-all.txt", ["ncu", "--query-metrics", "--chips", "gb20b", "--query-metrics-mode", "all"]),
    ("ncu-gb20b-pmsampling.txt", ["ncu", "--query-metrics", "--chips", "gb20b", "--query-metrics-collection", "pmsampling"]),
    ("ncu-gb20b-warpsampling.txt", ["ncu", "--query-metrics", "--chips", "gb20b", "--query-metrics-collection", "warpsampling"]),
    ("ncu-supported-chips.txt", ["ncu", "--list-chips"]),
    ("ncu-help.txt", ["ncu", "--help"]),
    ("nvidia-smi-fields.txt", ["nvidia-smi", "--help-query-gpu"]),
    ("nvidia-smi-current.csv", ["nvidia-smi", "--query-gpu=name,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,temperature.gpu,power.draw,clocks.current.sm,clocks.current.memory", "--format=csv"]),
]
results = []
for filename, command in queries:
    with (root / filename).open("w") as output, (root / (filename + ".stderr")).open("w") as error:
        process = subprocess.run(command, stdout=output, stderr=error, timeout=90)
    # Some ncu query errors return process status 0, so inspect the message too.
    with (root / filename).open() as output:
        first_lines = "".join(next(output, "") for _ in range(4))
    results.append({"command": command, "file": filename, "returncode": process.returncode,
                    "ncu_error": "==ERROR==" in first_lines})
    print(filename, process.returncode, "error" if results[-1]["ncu_error"] else "ok", flush=True)
(root / "commands.json").write_text(json.dumps(results, indent=2) + "\n")
subprocess.run([sys.executable, str(root / "build_inventory.py")], check=True)
