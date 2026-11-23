#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

static void mark(const char *text)
{
    if (write(STDERR_FILENO, text, std::strlen(text)) < 0)
        std::exit(2);
}

static void check(CUresult result)
{
    if (result != CUDA_SUCCESS) {
        std::fprintf(stderr, "CUDA error: %d\n", int(result));
        std::exit(1);
    }
}

static void dump_maps()
{
    int fd = open("/proc/self/maps", O_RDONLY | O_CLOEXEC);
    if (fd < 0)
        std::exit(2);
    char buffer[4096];
    ssize_t count;
    while ((count = read(fd, buffer, sizeof(buffer))) > 0) {
        ssize_t written = 0;
        while (written < count) {
            ssize_t result = write(STDERR_FILENO, buffer + written, count - written);
            if (result <= 0)
                std::exit(2);
            written += result;
        }
    }
    close(fd);
    if (count < 0)
        std::exit(2);
}

int main()
{
    mark("BEGIN CUDA_INIT\n");
    mark("BEGIN cuInit\n");
    check(cuInit(0));
    mark("END cuInit\n");
    CUdevice device;
    mark("BEGIN cuDeviceGet\n");
    check(cuDeviceGet(&device, 0));
    mark("END cuDeviceGet\n");
    CUcontext context;
    mark("BEGIN cuDevicePrimaryCtxRetain\n");
    check(cuDevicePrimaryCtxRetain(&context, device));
    mark("END cuDevicePrimaryCtxRetain\n");
    mark("BEGIN cuCtxSetCurrent\n");
    check(cuCtxSetCurrent(context));
    mark("END cuCtxSetCurrent\n");
    mark("END CUDA_INIT\n");

    mark("BEGIN MAPS_BEFORE\n");
    dump_maps();
    mark("END MAPS_BEFORE\n");

    const size_t size = 16 * 1024 * 1024;
    CUdeviceptr address = 0;
    mark("BEGIN cuMemAddressReserve\n");
    CUresult result = cuMemAddressReserve(&address, size, 0, 0, 0);
    mark("END cuMemAddressReserve\n");
    check(result);
    std::fprintf(stderr, "RESERVED_ADDRESS=0x%llx size=%zu\n", static_cast<unsigned long long>(address), size);

    mark("BEGIN MAPS_AFTER\n");
    dump_maps();
    mark("END MAPS_AFTER\n");

    mark("BEGIN cuMemAddressFree\n");
    result = cuMemAddressFree(address, size);
    mark("END cuMemAddressFree\n");
    check(result);
    mark("RESERVATION RELEASED; NO cuMemCreate OR GPU KERNELS\n");
    check(cuCtxSetCurrent(nullptr));
    check(cuDevicePrimaryCtxRelease(device));
    return 0;
}
