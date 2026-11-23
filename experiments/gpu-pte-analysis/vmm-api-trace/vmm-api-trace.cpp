#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

static void mark(const char *text)
{
    const size_t length = std::strlen(text);
    if (write(STDERR_FILENO, text, length) != static_cast<ssize_t>(length))
        std::exit(2);
}

static void check(CUresult result)
{
    if (result != CUDA_SUCCESS) {
        const char *name = nullptr;
        cuGetErrorName(result, &name);
        std::fprintf(stderr, "CUDA error: %d (%s)\n", int(result), name ? name : "unknown");
        std::exit(1);
    }
}

template <typename Function>
static void step(const char *name, Function function)
{
    char marker[128];
    std::snprintf(marker, sizeof(marker), "BEGIN %s\n", name);
    mark(marker);
    CUresult result = function();
    std::snprintf(marker, sizeof(marker), "END %s\n", name);
    mark(marker);
    check(result);
}

int main()
{
    mark("BEGIN CUDA_INIT\n");
    check(cuInit(0));
    CUdevice device;
    check(cuDeviceGet(&device, 0));
    CUcontext context;
    check(cuDevicePrimaryCtxRetain(&context, device));
    check(cuCtxSetCurrent(context));

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    size_t granularity = 0;
    check(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    const size_t requested = 16 * 1024 * 1024;
    const size_t size = ((requested + granularity - 1) / granularity) * granularity;
    std::fprintf(stderr, "granularity=%zu size=%zu\n", granularity, size);
    mark("END CUDA_INIT\n");

    CUdeviceptr address = 0;
    CUmemGenericAllocationHandle handle = 0;
    step("cuMemAddressReserve", [&] { return cuMemAddressReserve(&address, size, 0, 0, 0); });
    step("cuMemCreate", [&] { return cuMemCreate(&handle, size, &prop, 0); });
    step("cuMemMap", [&] { return cuMemMap(address, size, 0, handle, 0); });
    CUmemAccessDesc access = {};
    access.location = prop.location;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    step("cuMemSetAccess", [&] { return cuMemSetAccess(address, size, &access, 1); });

    // No CPU/GPU accesses to the buffer and no GPU kernels are launched.
    step("cuMemUnmap", [&] { return cuMemUnmap(address, size); });
    step("cuMemAddressFree", [&] { return cuMemAddressFree(address, size); });
    step("cuMemRelease", [&] { return cuMemRelease(handle); });
    mark("ALL TEST ALLOCATIONS RELEASED\n");
    check(cuCtxSetCurrent(nullptr));
    check(cuDevicePrimaryCtxRelease(device));
    return 0;
}
