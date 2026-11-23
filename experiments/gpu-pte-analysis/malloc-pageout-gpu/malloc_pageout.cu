#include <cuda_runtime.h>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include "../gpu-managed-fault-minimal/fault_monitor.h"

static void require(bool ok, const char *what)
{
    if (!ok) { std::perror(what); std::exit(2); }
}

static void cuda_check(cudaError_t status)
{
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(status));
        std::exit(3);
    }
}

__global__ void warmup() {}
__global__ void read_target(const volatile unsigned *data, unsigned *result)
{
    *result = data[0];
}
__global__ void verify_increment(volatile unsigned *data, size_t words, unsigned *result)
{
    unsigned errors = 0;
    for (size_t i = 0; i < words; ++i) {
        unsigned value = data[i];
        errors += value != 0x13570000u + i;
        data[i] = value + 1;
    }
    *result = errors;
}

struct Snapshot {
    std::vector<uint64_t> entries;
    std::vector<unsigned char> resident;
};

// Both interfaces inspect metadata; neither dereferences the tested allocation.
static Snapshot snapshot(int fd, void *data, size_t bytes, size_t page_size)
{
    Snapshot state{std::vector<uint64_t>(bytes / page_size),
                   std::vector<unsigned char>(bytes / page_size)};
    off_t offset = (reinterpret_cast<uintptr_t>(data) / page_size) * sizeof(uint64_t);
    size_t entry_bytes = state.entries.size() * sizeof(uint64_t);
    require(pread(fd, state.entries.data(), entry_bytes, offset) == ssize_t(entry_bytes), "pagemap");
    require(mincore(data, bytes, state.resident.data()) == 0, "mincore");
    return state;
}

static bool present(uint64_t entry) { return entry >> 63; }
static bool swapped(uint64_t entry) { return (entry >> 62) & 1; }

static void show(const char *stage, const Snapshot &state, size_t target)
{
    size_t mapped = 0, swap_entries = 0, cached = 0;
    for (size_t i = 0; i < state.entries.size(); ++i) {
        mapped += present(state.entries[i]);
        swap_entries += swapped(state.entries[i]);
        cached += bool(state.resident[i] & 1);
    }
    std::printf("PAGES %s total=%zu present=%zu swapped=%zu mincore_resident=%zu\n",
                stage, state.entries.size(), mapped, swap_entries, cached);
    std::printf("TARGET %s index=%zu present=%d swapped=%d mincore_resident=%d\n",
                stage, target, present(state.entries[target]), swapped(state.entries[target]),
                bool(state.resident[target] & 1));
}

static unsigned long long read_counter(const char *path, const char *name)
{
    FILE *fp = std::fopen(path, "r");
    require(fp != nullptr, path);
    char line[512], key[128];
    unsigned long long value = 0;
    bool found = false;
    while (std::fgets(line, sizeof(line), fp)) {
        if (std::sscanf(line, "%127s %llu", key, &value) == 2 && !std::strcmp(key, name)) {
            found = true;
            break;
        }
    }
    std::fclose(fp);
    require(found, name);
    return value;
}

static std::string cgroup_path()
{
    FILE *fp = std::fopen("/proc/self/cgroup", "r");
    require(fp != nullptr, "cgroup");
    char line[4096];
    std::string result;
    while (std::fgets(line, sizeof(line), fp)) {
        if (!std::strncmp(line, "0::/", 4)) {
            line[std::strcspn(line, "\n")] = 0;
            result = std::string("/sys/fs/cgroup") + (line + 3);
            break;
        }
    }
    std::fclose(fp);
    require(!result.empty(), "cgroup v2 path");
    return result;
}

static void reclaim_own_group(const std::string &group)
{
    // Only an explicitly created, dedicated experiment service is eligible.
    require(group.find("/malloc-pageout-gpu-") != std::string::npos &&
            group.size() >= 8 && group.compare(group.size() - 8, 8, ".service") == 0,
            "reclaim requires a dedicated malloc-pageout-gpu-*.service");
    std::string path = group + "/memory.reclaim";
    int fd = open(path.c_str(), O_WRONLY | O_CLOEXEC);
    require(fd >= 0, "open own memory.reclaim (Delegate=yes required)");
    const char request[] = "32M swappiness=max";
    ssize_t rc = write(fd, request, sizeof(request) - 1);
    int saved_errno = errno;
    close(fd);
    // EAGAIN means the requested total was not reclaimed; inspect actual pages.
    require(rc == sizeof(request) - 1 || (rc < 0 && saved_errno == EAGAIN), "memory.reclaim");
    std::printf("RECLAIM requested=32M swappiness=max rc=%zd errno=%d\n", rc, rc < 0 ? saved_errno : 0);
}

int main(int argc, char **argv)
{
    setvbuf(stdout, nullptr, _IOLBF, 0);
    bool pageout = true, warm = false, reclaim = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "resident")) pageout = false;
        else if (!std::strcmp(argv[i], "pageout")) pageout = true;
        else if (!std::strcmp(argv[i], "warm")) warm = true;
        else if (!std::strcmp(argv[i], "reclaim")) reclaim = true;
        else {
            std::fprintf(stderr, "Usage: %s [resident|pageout] [warm] [reclaim]\n", argv[0]);
            return 2;
        }
    }
    require(!reclaim || pageout, "reclaim requires pageout");
    cuda_check(cudaSetDevice(0));
    int pageable = 0, ats = 0;
    cuda_check(cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, 0));
    cuda_check(cudaDeviceGetAttribute(&ats, cudaDevAttrPageableMemoryAccessUsesHostPageTables, 0));
    std::printf("ATTR pageableMemoryAccess=%d usesHostPageTables=%d\n", pageable, ats);
    if (!pageable || !ats) return 3;
    warmup<<<1, 1>>>();
    cuda_check(cudaGetLastError());
    cuda_check(cudaDeviceSynchronize());
    unsigned *result;
    cuda_check(cudaMalloc(&result, sizeof(unsigned)));
    GpuFaultMonitor monitor;

    size_t page_size = sysconf(_SC_PAGESIZE);
    size_t bytes = 16 * 1024 * 1024;
    // Keep the allocator's header and neighboring allocations outside PAGEOUT.
    // This is an ordinary malloc allocation; do not register or pin it with CUDA.
    void *allocation = std::malloc(bytes + 2 * page_size);
    require(allocation != nullptr, "malloc");
    uintptr_t aligned = (reinterpret_cast<uintptr_t>(allocation) + page_size - 1) & ~(page_size - 1);
    auto *data = reinterpret_cast<unsigned *>(aligned);
    require(madvise(data, bytes, MADV_NOHUGEPAGE) == 0, "MADV_NOHUGEPAGE");
    size_t words_per_page = page_size / sizeof(unsigned);
    for (size_t p = 0; p < bytes / page_size; ++p)
        for (size_t i = 0; i < words_per_page; ++i)
            data[p * words_per_page + i] = 0x13570000u + i;
    std::printf("MODE pageout=%d warm=%d reclaim=%d\n", pageout, warm, reclaim);
    std::printf("ALLOCATION malloc=%p aligned_interior=%p bytes=%zu page_size=%zu\n",
                allocation, data, bytes, page_size);
    if (warm) {
        read_target<<<1, 1>>>(data, result);
        cuda_check(cudaGetLastError());
        cuda_check(cudaDeviceSynchronize());
        unsigned value = 0;
        cuda_check(cudaMemcpy(&value, result, sizeof(value), cudaMemcpyDeviceToHost));
        require(value == 0x13570000u, "prior GPU read mismatch");
        std::puts("PRIOR_GPU_READ PASS target_index=0");
    }

    int fd = open("/proc/self/pagemap", O_RDONLY | O_CLOEXEC);
    require(fd >= 0, "open pagemap");
    std::string group = cgroup_path();
    std::string memory_stat = group + "/memory.stat";
    std::printf("CGROUP %s\n", group.c_str());
    std::printf("BASELINE_GPU_FAULTS=%u\n", monitor.report(data, bytes));
    show("before_pageout", snapshot(fd, data, bytes, page_size), 0);
    if (pageout) {
        require(madvise(data, bytes, MADV_PAGEOUT) == 0, "MADV_PAGEOUT");
        // Allow asynchronous swap writes to finish, then optionally evict cache
        // using only this dedicated service's cgroup, with no global pressure.
        usleep(500000);
        if (reclaim) reclaim_own_group(group);
    }

    Snapshot before = snapshot(fd, data, bytes, page_size);
    // Test the same page in cold and prior-GPU-access runs.
    size_t target_index = 0;
    show("before_gpu", before, target_index);
    uint64_t entry = before.entries[target_index];
    bool ready = pageout ? !present(entry) && swapped(entry) : present(entry) && !swapped(entry);
    if (!ready) { std::puts("RESULT INCONCLUSIVE pre-GPU page state not confirmed"); return 5; }
    if (reclaim && (before.resident[target_index] & 1)) {
        std::puts("RESULT INCONCLUSIVE target still in swap cache");
        return 5;
    }
    unsigned *target = data + target_index * words_per_page;
    auto swapin_before = read_counter("/proc/vmstat", "pswpin");
    auto major_before = read_counter(memory_stat.c_str(), "pgmajfault");
    timespec time_before{}, time_after{};
    clock_gettime(CLOCK_MONOTONIC, &time_before);
    // No application CPU target-data access since initialization. Pass the
    // malloc pointer directly to the GPU, without cudaHostRegister/prefetch/copy.
    verify_increment<<<1, 1>>>(target, words_per_page, result);
    cuda_check(cudaGetLastError());
    cuda_check(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &time_after);
    Snapshot after = snapshot(fd, data, bytes, page_size);
    auto swapin_after = read_counter("/proc/vmstat", "pswpin");
    auto major_after = read_counter(memory_stat.c_str(), "pgmajfault");
    show("after_gpu_before_cpu_read", after, target_index);
    unsigned faults = monitor.report(target, page_size);
    std::printf("TARGET_GPU_FAULTS=%u\n", faults);
    double elapsed_ms = (time_after.tv_sec - time_before.tv_sec) * 1000.0 +
                        (time_after.tv_nsec - time_before.tv_nsec) / 1e6;
    std::printf("COUNTERS global_pswpin_delta=%llu cgroup_pgmajfault_delta=%llu gpu_phase_ms=%.3f\n",
                swapin_after - swapin_before, major_after - major_before, elapsed_ms);
    unsigned gpu_errors = 0, cpu_errors = 0;
    cuda_check(cudaMemcpy(&gpu_errors, result, sizeof(gpu_errors), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < words_per_page; ++i)
        cpu_errors += target[i] != 0x13570001u + i;
    bool pass = present(after.entries[target_index]) && !swapped(after.entries[target_index]) &&
                !gpu_errors && !cpu_errors;
    std::printf("VERIFY gpu_errors=%u cpu_errors=%u words=%zu\n", gpu_errors, cpu_errors, words_per_page);
    std::printf("RECOVERY %s\n", !pageout ? "resident_control" :
                (before.resident[target_index] & 1) ? "swap_cache" : "nonresident_swap");
    close(fd);
    cuda_check(cudaFree(result));
    std::free(allocation);
    std::printf("RESULT %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 6;
}
