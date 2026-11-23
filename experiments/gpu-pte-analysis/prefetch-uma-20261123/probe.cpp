#include <cuda_runtime_api.h>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>
#include <vector>

static void check(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess) {
        std::fprintf(stderr, "ERROR %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(1);
    }
}

static unsigned long long present_pages(const void *pointer, size_t length)
{
    const size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    const uintptr_t start = reinterpret_cast<uintptr_t>(pointer) & ~(page_size - 1);
    const uintptr_t end = (reinterpret_cast<uintptr_t>(pointer) + length + page_size - 1) & ~(page_size - 1);
    std::vector<uint64_t> entries((end - start) / page_size);
    const int fd = open("/proc/self/pagemap", O_RDONLY);
    if (fd < 0) {
        std::perror("open pagemap");
        std::exit(1);
    }
    size_t done = 0;
    const size_t total = entries.size() * sizeof(uint64_t);
    while (done < total) {
        const ssize_t count = pread(fd, reinterpret_cast<char *>(entries.data()) + done,
                                    total - done, (start / page_size) * sizeof(uint64_t) + done);
        if (count < 0 && errno == EINTR)
            continue;
        if (count <= 0) {
            std::perror("pread pagemap");
            std::exit(1);
        }
        done += static_cast<size_t>(count);
    }
    close(fd);
    unsigned long long present = 0;
    for (uint64_t entry : entries)
        present += (entry >> 63) & 1;
    return present;
}

static void run_case(const char *name, bool cpu_touch, bool destination_gpu, cudaStream_t stream)
{
    constexpr size_t length = 8 * 1024 * 1024;
    const size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    std::fprintf(stderr, "PHASE BEGIN %s allocation\n", name);
    void *pointer = nullptr;
    check(cudaMallocManaged(&pointer, length), "cudaMallocManaged");
    const auto after_allocation = present_pages(pointer, length);

    volatile unsigned char *bytes = static_cast<volatile unsigned char *>(pointer);
    if (cpu_touch) {
        std::fprintf(stderr, "PHASE %s CPU_touch\n", name);
        for (size_t offset = 0; offset < length; offset += page_size)
            bytes[offset] = 0x5a;
        bytes[length - 1] = 0xa5;
    }
    const auto before_prefetch = present_pages(pointer, length);
    cudaMemLocation location{};
    location.type = destination_gpu ? cudaMemLocationTypeDevice : cudaMemLocationTypeHost;
    location.id = 0;
    std::fprintf(stderr, "PHASE %s prefetch_%s\n", name, destination_gpu ? "GPU" : "CPU");
    check(cudaMemPrefetchAsync(pointer, length, location, 0, stream), "cudaMemPrefetchAsync");
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    const auto after_prefetch = present_pages(pointer, length);

    bool data_check = true;
    if (cpu_touch) {
        for (size_t offset = 0; offset < length; offset += page_size)
            data_check &= bytes[offset] == 0x5a;
        data_check &= bytes[length - 1] == 0xa5;
    }
    std::printf("{\"case\":\"%s\",\"bytes\":%zu,\"pages\":%zu,"
                "\"CPU_present_after_allocation\":%llu,\"CPU_present_before_prefetch\":%llu,"
                "\"CPU_present_after_prefetch_sync\":%llu,\"CPU_initialized_data_check\":%s}\n",
                name, length, length / page_size, after_allocation, before_prefetch,
                after_prefetch, cpu_touch ? (data_check ? "true" : "false") : "null");
    std::fprintf(stderr, "PHASE %s free\n", name);
    check(cudaFree(pointer), "cudaFree");
    std::fprintf(stderr, "PHASE END %s\n", name);
    if (!data_check)
        std::exit(1);
}

int main()
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    check(cudaSetDevice(0), "cudaSetDevice");
    check(cudaFree(nullptr), "CUDA initialization");
    for (const auto attribute : {cudaDevAttrIntegrated, cudaDevAttrManagedMemory,
                                cudaDevAttrConcurrentManagedAccess, cudaDevAttrPageableMemoryAccess,
                                cudaDevAttrPageableMemoryAccessUsesHostPageTables}) {
        int value = 0;
        check(cudaDeviceGetAttribute(&value, attribute, 0), "cudaDeviceGetAttribute");
        std::printf("{\"device_attribute\":%d,\"value\":%d}\n", static_cast<int>(attribute), value);
    }
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    run_case("untouched_to_GPU", false, true, stream);
    run_case("untouched_to_CPU", false, false, stream);
    run_case("CPU_touched_to_GPU", true, true, stream);
    run_case("CPU_touched_to_CPU", true, false, stream);
    check(cudaStreamDestroy(stream), "cudaStreamDestroy");
}
