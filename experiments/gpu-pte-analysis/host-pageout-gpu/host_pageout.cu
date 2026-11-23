#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include "../gpu-managed-fault-minimal/fault_monitor.h"

static void check(cudaError_t status)
{
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(status));
        std::exit(1);
    }
}

static void os_check(bool success, const char *operation)
{
    if (!success) { std::perror(operation); std::exit(2); }
}

__global__ void warmup() {}
__global__ void read_head(const unsigned *data, unsigned *result) { *result = data[0]; }
__global__ void verify_and_increment(unsigned *data, size_t count, unsigned *result)
{
    unsigned errors = 0;
    for (size_t i = 0; i < count; ++i) {
        unsigned value = data[i];
        errors += value != 0x13570000u + i;
        data[i] = value + 1;
    }
    *result = errors;
}

struct PageState {
    bool present, swapped, resident;
};

// These operations inspect metadata without reading the target data.
static PageState page_state(int fd, void *data, size_t page_size)
{
    uint64_t entry = 0;
    off_t offset = (reinterpret_cast<uintptr_t>(data) / page_size) * sizeof(entry);
    os_check(pread(fd, &entry, sizeof(entry), offset) == sizeof(entry), "pagemap");
    unsigned char resident = 0;
    os_check(mincore(data, page_size, &resident) == 0, "mincore");
    return {bool(entry >> 63), bool((entry >> 62) & 1), bool(resident & 1)};
}

static void show(const char *stage, PageState state)
{
    std::printf("PAGE %s present=%d swapped=%d mincore_resident=%d\n",
                 stage, state.present, state.swapped, state.resident);
}

int main(int argc, char **argv)
{
    bool registered = false, warm_gpu = false, pageout = true;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "registered")) registered = true;
        else if (!std::strcmp(argv[i], "plain")) registered = false;
        else if (!std::strcmp(argv[i], "warm")) warm_gpu = true;
        else if (!std::strcmp(argv[i], "resident")) pageout = false;
        else {
            std::fprintf(stderr, "Usage: %s [plain|registered] [warm] [resident]\n", argv[0]);
            return 2;
        }
    }
    std::printf("MODE %s prior_gpu_access=%d pageout=%d\n", registered ? "registered" : "plain", warm_gpu, pageout);
    check(cudaSetDeviceFlags(cudaDeviceMapHost));
    check(cudaSetDevice(0));
    int pageable = 0, host_tables = 0;
    check(cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, 0));
    check(cudaDeviceGetAttribute(&host_tables, cudaDevAttrPageableMemoryAccessUsesHostPageTables, 0));
    std::printf("ATTR pageableMemoryAccess=%d usesHostPageTables=%d\n", pageable, host_tables);
    if (!pageable || !host_tables) return 3; // This probe is for this Spark's configuration.
    warmup<<<1, 1>>>();
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());

    unsigned *device_result = nullptr;
    check(cudaMalloc(&device_result, sizeof(unsigned)));
    GpuFaultMonitor monitor;
    size_t bytes = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    auto *data = static_cast<unsigned *>(mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    os_check(data != MAP_FAILED, "mmap");
    os_check(madvise(data, bytes, MADV_NOHUGEPAGE) == 0, "MADV_NOHUGEPAGE");
    size_t count = bytes / sizeof(unsigned);
    for (size_t i = 0; i < count; ++i) data[i] = 0x13570000u + i;
    unsigned *gpu_data = data;
    if (registered) {
        check(cudaHostRegister(data, bytes, cudaHostRegisterMapped));
        check(cudaHostGetDevicePointer(&gpu_data, data, 0));
    }
    std::printf("ALLOCATION cpu=%p gpu=%p bytes=%zu\n", static_cast<void *>(data),
                 static_cast<void *>(gpu_data), bytes);

    if (warm_gpu) {
        read_head<<<1, 1>>>(gpu_data, device_result);
        check(cudaGetLastError());
        check(cudaDeviceSynchronize());
        unsigned value = 0;
        check(cudaMemcpy(&value, device_result, sizeof(value), cudaMemcpyDeviceToHost));
        if (value != 0x13570000u) return 4;
        std::puts("GPU_BEFORE_PAGEOUT PASS");
    }
    std::printf("BASELINE_GPU_FAULTS=%u\n", monitor.report(gpu_data, bytes));
    int fd = open("/proc/self/pagemap", O_RDONLY | O_CLOEXEC);
    os_check(fd >= 0, "open pagemap");
    show("before_pageout", page_state(fd, data, bytes));
    if (pageout) os_check(madvise(data, bytes, MADV_PAGEOUT) == 0, "MADV_PAGEOUT");

    PageState before{};
    for (unsigned attempt = 0; attempt < 200; ++attempt) {
        before = page_state(fd, data, bytes);
        if (!pageout || (before.swapped && !before.present)) break;
        usleep(10000);
    }
    show(pageout ? "after_pageout_before_gpu" : "resident_before_gpu", before);
    bool ready = pageout ? before.swapped && !before.present : before.present && !before.swapped;
    if (!ready) {
        std::fprintf(stderr, "INCONCLUSIVE: expected pre-GPU pagemap state not confirmed\n");
        return 5;
    }
    if (pageout && before.resident)
        std::puts("NOTE swap entry confirmed; data still resident in swap cache, disk-read recovery not tested");

    // No CPU target-data access after initialization until the GPU completes.
    verify_and_increment<<<1, 1>>>(gpu_data, count, device_result);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
    std::printf("%s_GPU_FAULTS=%u\n", pageout ? "AFTER_PAGEOUT" : "RESIDENT", monitor.report(gpu_data, bytes));
    PageState after = page_state(fd, data, bytes);
    show("after_gpu_before_cpu_read", after);
    unsigned gpu_errors = 0;
    check(cudaMemcpy(&gpu_errors, device_result, sizeof(gpu_errors), cudaMemcpyDeviceToHost));
    unsigned cpu_errors = 0;
    for (size_t i = 0; i < count; ++i) cpu_errors += data[i] != 0x13570001u + i;
    bool pass = after.present && !after.swapped && !gpu_errors && !cpu_errors;
    std::printf("VERIFY gpu_errors=%u cpu_errors=%u words=%zu\n", gpu_errors, cpu_errors, count);
    if (registered) check(cudaHostUnregister(data));
    check(cudaFree(device_result));
    close(fd);
    os_check(munmap(data, bytes) == 0, "munmap");
    std::printf("RESULT %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 6;
}
