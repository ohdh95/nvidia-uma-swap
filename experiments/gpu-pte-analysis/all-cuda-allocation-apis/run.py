import subprocess, pathlib, os, json, re
base = pathlib.Path(__file__).resolve().parent
cases = [
 "cudaMalloc", "cudaMallocPitch", "cudaMalloc3D", "cudaMallocManaged", "cudaMallocManagedHost",
 "cudaMallocHost", "cudaHostAlloc", "cudaHostAllocMapped", "cudaHostAllocWC", "cudaHostRegister",
 "cudaMallocArray", "cudaMalloc3DArray", "cudaMallocMipmappedArray", "cudaMallocArraySparse", "cudaMallocMipmappedArraySparse",
 "cudaMallocAsync", "cudaPoolDevice", "cudaPoolHost", "cudaPoolHostNuma", "cudaPoolManaged",
 "cuMemAlloc", "cuMemAllocPitch", "cuMemAllocManaged", "cuMemAllocHost", "cuMemHostAlloc", "cuMemHostRegister",
 "cuArrayCreate", "cuArray3DCreate", "cuArray3DCreateSparse", "cuMipmappedArrayCreate",
 "cuMemCreateDevice", "cuMemCreateHost", "cuMemCreateHostNuma", "cuMemCreateTilePool",
 "cuMemAllocAsync", "cuPoolDevice", "cuPoolHost", "cuPoolHostNuma", "cuPoolManaged", "cudaGraphMemory", "cuGraphMemory"
]
results=[]
for case in cases:
    trace=base/(case+".strace")
    cmd=["strace", "-f", "-yy", "-tt", "-T", "-s", "512", "-e", "trace=ioctl,mmap,munmap,mlock,mlock2,munlock,madvise,write", "-o", str(trace), "env", "LD_PRELOAD=/tmp/dgx-rm-observe-all.so", "/tmp/dgx-all-cuda-allocation-probe", case]
    try:
        p=subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
        log=p.stdout+p.stderr
        (base/(case+".log")).write_text(log)
        section=log.split("BEGIN API "+case+"\n",1)[-1]
        (base/(case+".section.log")).write_text(section)
        allocations=[line for line in section.splitlines() if line.startswith("RM_")]
        status=[line for line in section.splitlines() if line.startswith("RESULT ")]
        lines=trace.read_text().splitlines()
        in_api=False; selected=[]
        for line in lines:
            if 'BEGIN API '+case+'\\n' in line: in_api=True
            if in_api: selected.append(line)
            if in_api and 'END API '+case+'\\n"' in line: break
        (base/(case+".section.strace")).write_text("\n".join(selected)+"\n")
        row={"case":case,"exit_code":p.returncode,"result":status,"rm_allocations":allocations,"section_trace_lines":len(selected),"cpu_mapping":[l for l in section.splitlines() if l.startswith(("VMA ","SMAPS "))]}
    except subprocess.TimeoutExpired:
        row={"case":case,"exit_code":124,"result":["TIMEOUT"]}
    results.append(row)
    (base/"observations.json").write_text(json.dumps(results,indent=2))
    print(case, "exit="+str(row["exit_code"]), "RM_allocs="+str(len(row.get("rm_allocations",[]))), "; ".join(row["result"]), flush=True)
