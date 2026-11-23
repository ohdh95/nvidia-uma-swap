#include <cuda_runtime_api.h>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

static bool check(cudaError_t status, const char *operation)
{
    if (status == cudaSuccess)
        return true;
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
    return false;
}

int main()
{
    if (!check(cudaSetDevice(0), "cudaSetDevice") ||
        !check(cudaFree(nullptr), "CUDA initialization"))
        return 1;

    void *allocation = nullptr;
    constexpr size_t allocation_size = 16 * 1024 * 1024;
    if (!check(cudaMalloc(&allocation, allocation_size), "cudaMalloc"))
        return 1;

    std::printf("cudaMalloc: pointer=%p size=%zu\n", allocation, allocation_size);
    std::printf("No CPU/GPU access to allocation contents.\n");

    std::ifstream smaps("/proc/self/smaps");
    const uintptr_t target = reinterpret_cast<uintptr_t>(allocation);
    bool found = false, saw_flags = false;
    bool locked = false, special = false;
    std::string line;
    while (std::getline(smaps, line)) {
        unsigned long long start, end;
        char permissions[5];
        if (std::sscanf(line.c_str(), "%llx-%llx %4s", &start, &end, permissions) == 3) {
            if (found)
                break;
            found = target >= start && target < end;
        }
        if (!found)
            continue;

        std::printf("%s\n", line.c_str());
        if (line.rfind("VmFlags:", 0) == 0) {
            saw_flags = true;
            std::istringstream flags(line.substr(8));
            std::string flag;
            while (flags >> flag) {
                locked |= flag == "lo";
                special |= flag == "io" || flag == "de" || flag == "pf" || flag == "mm";
            }
        }
    }

    if (found && saw_flags) {
        std::printf("VM_LOCKED=%d VM_SPECIAL=%d\n", locked, special);
        std::printf("(vm_flags & (VM_LOCKED | VM_SPECIAL)) == VM_LOCKED: %s\n",
                    locked && !special ? "true" : "false");
    }
    else
        std::fprintf(stderr, "Could not find the allocation's CPU VMA flags.\n");

    const bool freed = check(cudaFree(allocation), "cudaFree");
    if (freed)
        std::printf("cudaFree: success\n");
    return found && saw_flags && freed ? 0 : 1;
}
