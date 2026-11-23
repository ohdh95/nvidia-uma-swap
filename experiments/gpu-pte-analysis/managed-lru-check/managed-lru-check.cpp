#include <cuda_runtime_api.h>
#include <cstdint>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
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
    if (!check(cudaMallocManaged(&allocation, allocation_size), "cudaMallocManaged"))
        return 1;

    std::printf("cudaMallocManaged: pointer=%p size=%zu\n", allocation, allocation_size);
    volatile unsigned char *bytes = static_cast<volatile unsigned char *>(allocation);
    for (size_t offset = 0; offset < allocation_size; offset += 4096)
        bytes[offset] = 0x5a;
    bytes[allocation_size - 1] = 0xa5;
    std::printf("CPU wrote one byte per 4KiB page; first=%u last=%u. No GPU data access.\n", bytes[0], bytes[allocation_size - 1]);

    int flags_fd = open("/proc/kpageflags", O_RDONLY);
    if (flags_fd < 0)
        std::printf("Direct kpageflags observation unavailable: %s\n", std::strerror(errno));
    else {
        std::printf("kpageflags opened; PFN resolution is still required.\n");
        close(flags_fd);
    }

    int pagemap_fd = open("/proc/self/pagemap", O_RDONLY);
    if (pagemap_fd >= 0) {
        uint64_t entry = 0;
        const off_t offset = (reinterpret_cast<uintptr_t>(allocation) / 4096) * sizeof(entry);
        if (pread(pagemap_fd, &entry, sizeof(entry), offset) == sizeof(entry)) {
            const uint64_t pfn = entry & ((1ULL << 55) - 1);
            std::printf("pagemap present=%llu swapped=%llu pfn_field=%llu (zero may indicate masked PFN)\n",
                        static_cast<unsigned long long>((entry >> 63) & 1),
                        static_cast<unsigned long long>((entry >> 62) & 1),
                        static_cast<unsigned long long>(pfn));
        }
        close(pagemap_fd);
    }

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
