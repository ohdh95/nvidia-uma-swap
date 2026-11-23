#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/utsname.h>
#include "uvm-event-monitor.h"

static void runtime_ok(cudaError_t status, const char *name)
{
    if (status == cudaSuccess) return;
    std::fprintf(stderr, "UNEXPECTED_CUDA_ERROR %s %d %s\n", name, int(status), cudaGetErrorString(status));
    std::exit(2);
}

static void driver_ok(CUresult status, const char *name)
{
    if (status == CUDA_SUCCESS) return;
    const char *message = nullptr;
    cuGetErrorString(status, &message);
    std::fprintf(stderr, "UNEXPECTED_DRIVER_ERROR %s %d %s\n", name, int(status), message ? message : "unknown");
    std::exit(2);
}

__global__ void warmup() {}
__global__ void store_one(volatile unsigned *data, unsigned value) { *data = value; }

int main(int argc, char **argv)
{
    if (argc != 3) return 2;
    const char *mode = argv[1];
    bool create = !std::strcmp(mode, "created_unmapped") || !std::strcmp(mode, "mapped");
    bool mapped = !std::strcmp(mode, "mapped");
    if (std::strcmp(mode, "reserved_only") && !create) return 2;
    std::fprintf(stderr, "MODE %s\n", mode);
    utsname info{};
    if (uname(&info) == 0) std::fprintf(stderr, "KERNEL %s\n", info.release);
    runtime_ok(cudaSetDevice(0), "cudaSetDevice");
    cudaDeviceProp device{};
    runtime_ok(cudaGetDeviceProperties(&device, 0), "cudaGetDeviceProperties");
    std::fprintf(stderr, "GPU %s cc=%d.%d\n", device.name, device.major, device.minor);
    warmup<<<1, 1>>>();
    runtime_ok(cudaGetLastError(), "warmup launch");
    runtime_ok(cudaDeviceSynchronize(), "warmup sync");
    UvmEventMonitor monitor;
    if (!monitor.start(argv[2])) return 3;

    CUmemAllocationProp property{};
    property.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    property.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    property.location.id = 0;
    size_t granularity = 0;
    driver_ok(cuMemGetAllocationGranularity(&granularity, &property,
              CU_MEM_ALLOC_GRANULARITY_MINIMUM), "cuMemGetAllocationGranularity");
    const size_t bytes = ((16 * 1024 * 1024 + granularity - 1) / granularity) * granularity;
    CUdeviceptr address = 0;
    driver_ok(cuMemAddressReserve(&address, bytes, granularity, 0, 0), "cuMemAddressReserve");
    std::fprintf(stderr, "RESERVATION base=0x%llx bytes=%zu granularity=%zu\n",
                 static_cast<unsigned long long>(address), bytes, granularity);
    CUmemGenericAllocationHandle handle = 0;
    if (create) {
        driver_ok(cuMemCreate(&handle, bytes, &property, 0), "cuMemCreate");
        std::fprintf(stderr, "PHYSICAL_ALLOCATION_CREATED\n");
    }
    if (mapped) {
        driver_ok(cuMemMap(address, bytes, 0, handle, 0), "cuMemMap");
        CUmemAccessDesc access{};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = 0;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        driver_ok(cuMemSetAccess(address, bytes, &access, 1), "cuMemSetAccess");
        std::fprintf(stderr, "GPU_MAPPING_AND_ACCESS_CONFIGURED\n");
    }
    void *target = reinterpret_cast<void *>(address);
    monitor.drain("before_gpu_access", target, bytes);
    constexpr unsigned expected = 0x13572468;
    store_one<<<1, 1>>>(reinterpret_cast<volatile unsigned *>(address), expected);
    cudaError_t launch = cudaGetLastError();
    std::fprintf(stderr, "GPU_LAUNCH status=%d name=%s\n", int(launch), cudaGetErrorName(launch));
    cudaError_t completion = cudaDeviceSynchronize();
    std::fprintf(stderr, "GPU_SYNC status=%d name=%s message=%s\n",
                 int(completion), cudaGetErrorName(completion), cudaGetErrorString(completion));
    monitor.drain("gpu_first_touch", target, bytes);
    bool pass;
    if (mapped) {
        unsigned value = 0;
        runtime_ok(launch, "mapped launch");
        runtime_ok(completion, "mapped sync");
        runtime_ok(cudaMemcpy(&value, target, sizeof(value), cudaMemcpyDeviceToHost), "copy verification value");
        pass = value == expected;
        std::fprintf(stderr, "VERIFY value=0x%x expected=0x%x\n", value, expected);
        driver_ok(cuMemUnmap(address, bytes), "cuMemUnmap");
        driver_ok(cuMemRelease(handle), "cuMemRelease");
        driver_ok(cuMemAddressFree(address, bytes), "cuMemAddressFree");
    }
    else {
        pass = launch == cudaSuccess && completion == cudaErrorIllegalAddress;
        // The expected illegal access poisons this process's CUDA context.
        // Leave allocation/context cleanup to normal process teardown.
    }
    std::fprintf(stderr, "RESULT %s %s\n", pass ? "PASS" : "FAIL", mode);
    return pass ? 0 : 1;
}
