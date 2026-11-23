#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <cstdio>
#include <cstdlib>
#include <sys/utsname.h>
#ifdef ENABLE_UVM_TOOLS
#include "uvm-event-monitor.h"
#endif

static void check(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(1);
    }
}

__global__ void warmup() {}

__global__ void touch_pages(unsigned *data, size_t page_count, unsigned value)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < page_count)
        data[index * (4096 / sizeof(unsigned))] = value + static_cast<unsigned>(index);
}

int main(int argc, char **argv)
{
    utsname info{};
    if (uname(&info) == 0)
        std::fprintf(stderr, "KERNEL %s\n", info.release);
    check(cudaSetDevice(0), "cudaSetDevice");
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    std::fprintf(stderr, "GPU %s cc=%d.%d\n", prop.name, prop.major, prop.minor);
    warmup<<<1, 1>>>();
    check(cudaGetLastError(), "warmup launch");
    check(cudaDeviceSynchronize(), "warmup sync");

#ifdef ENABLE_UVM_TOOLS
    UvmEventMonitor monitor;
    if (argc != 2 || !monitor.start(argv[1])) {
        std::fprintf(stderr, "Could not subscribe to UVM events; pass the output CSV path.\n");
        return 3;
    }
#endif

    constexpr size_t bytes = 16 * 1024 * 1024;
    constexpr size_t pages = bytes / 4096;
    unsigned *data = nullptr;
    nvtxRangePushA("MANAGED_ALLOC_NO_DATA_ACCESS");
    check(cudaMallocManaged(&data, bytes), "cudaMallocManaged");
    nvtxRangePop();
    std::fprintf(stderr, "ALLOCATION base=%p end=%p bytes=%zu pages=%zu\n",
                 static_cast<void *>(data), reinterpret_cast<char *>(data) + bytes, bytes, pages);
#ifdef ENABLE_UVM_TOOLS
    monitor.drain("after_alloc_before_access", data, bytes);
#endif

    // Establish CPU mappings before the first GPU access to this allocation.
    volatile unsigned *cpu = data;
    for (size_t index=0; index<pages; ++index) cpu[index*(4096/sizeof(unsigned))]=static_cast<unsigned>(index);
    std::fprintf(stderr, "CPU_FIRST_TOUCH_COMPLETE\n");
#ifdef ENABLE_UVM_TOOLS
    monitor.drain("cpu_first_touch_before_gpu", data, bytes);
#endif

    constexpr unsigned first_value = 0x12340000;
    constexpr unsigned second_value = 0x24680000;
    nvtxRangePushA("GPU_FIRST_TOUCH_AFTER_CPU_TOUCH");
    touch_pages<<<(pages + 127) / 128, 128>>>(data, pages, first_value);
    check(cudaGetLastError(), "first GPU launch");
    check(cudaDeviceSynchronize(), "first GPU sync");
    nvtxRangePop();
    std::fprintf(stderr, "FIRST_GPU_TOUCH_COMPLETE\n");
#ifdef ENABLE_UVM_TOOLS
    monitor.drain("gpu_first_touch", data, bytes);
#endif

    nvtxRangePushA("GPU_SECOND_TOUCH_SAME_ALLOCATION");
    touch_pages<<<(pages + 127) / 128, 128>>>(data, pages, second_value);
    check(cudaGetLastError(), "second GPU launch");
    check(cudaDeviceSynchronize(), "second GPU sync");
    nvtxRangePop();
    std::fprintf(stderr, "SECOND_GPU_TOUCH_COMPLETE\n");
#ifdef ENABLE_UVM_TOOLS
    monitor.drain("gpu_second_touch", data, bytes);
#endif

    nvtxRangePushA("CPU_VERIFY_AFTER_BOTH_GPU_TOUCHES");
    size_t mismatches = 0;
    for (size_t index = 0; index < pages; ++index) {
        if (data[index * (4096 / sizeof(unsigned))] != second_value + index)
            ++mismatches;
    }
    nvtxRangePop();
    std::fprintf(stderr, "VERIFY pages=%zu mismatches=%zu\n", pages, mismatches);
#ifdef ENABLE_UVM_TOOLS
    monitor.drain("cpu_verify", data, bytes);
#endif
    check(cudaFree(data), "cudaFree");
    std::fprintf(stderr, "RESULT %s\n", mismatches ? "FAIL" : "PASS");
    return mismatches ? 2 : 0;
}
