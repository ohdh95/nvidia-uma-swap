#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <unistd.h>

static void begin(const char *name, size_t bytes = 0)
{
    dprintf(2, "PHASE_BEGIN %s requested_bytes=%zu\n", name, bytes);
}
static void end(const char *name) { dprintf(2, "PHASE_END %s\n", name); }
static void driver(CUresult status, const char *call)
{
    if (status == CUDA_SUCCESS) return;
    const char *message = nullptr;
    cuGetErrorString(status, &message);
    dprintf(2, "FAIL %s status=%d message=%s\n", call, int(status), message ? message : "unknown");
    std::exit(1);
}
static void runtime(cudaError_t status, const char *call)
{
    if (status == cudaSuccess) return;
    dprintf(2, "FAIL %s status=%d message=%s\n", call, int(status), cudaGetErrorString(status));
    std::exit(1);
}
#define DR(call) driver((call), #call)
#define RT(call) runtime((call), #call)

int main(int argc, char **argv)
{
    if (argc != 2) return 2;
    const char *mode = argv[1];
    dprintf(2, "PROBE mode=%s pid=%d\n", mode, int(getpid()));
    if (std::strcmp(mode, "cold") == 0) {
        void *pointer = nullptr;
        begin("cold_cudaMalloc", 16U << 20);
        RT(cudaMalloc(&pointer, 16U << 20));
        dprintf(2, "RETURN cold_cudaMalloc pointer=%p\n", pointer);
        end("cold_cudaMalloc");
        begin("cudaFree"); RT(cudaFree(pointer)); end("cudaFree");
        begin("cudaDeviceReset"); RT(cudaDeviceReset()); end("cudaDeviceReset");
    }
    else {
        begin("cuInit"); DR(cuInit(0)); end("cuInit");
        if (std::strcmp(mode, "init-only") == 0) {
            dprintf(2, "RESULT PASS %s\n", mode);
            return 0;
        }
        if (std::strcmp(mode, "driver") == 0) {
            CUdevice device;
            CUcontext context;
            begin("cuDeviceGet"); DR(cuDeviceGet(&device, 0)); end("cuDeviceGet");
            begin("cuDevicePrimaryCtxRetain"); DR(cuDevicePrimaryCtxRetain(&context, device)); end("cuDevicePrimaryCtxRetain");
            begin("cuCtxSetCurrent"); DR(cuCtxSetCurrent(context)); end("cuCtxSetCurrent");
            CUdeviceptr pointer;
            begin("cuMemAlloc", 16U << 20); DR(cuMemAlloc(&pointer, 16U << 20));
            dprintf(2, "RETURN cuMemAlloc pointer=0x%llx\n", (unsigned long long)pointer);
            end("cuMemAlloc");
            begin("cuMemFree"); DR(cuMemFree(pointer)); end("cuMemFree");
            begin("cuDevicePrimaryCtxRelease"); DR(cuDevicePrimaryCtxRelease(device)); end("cuDevicePrimaryCtxRelease");
        }
        else if (std::strcmp(mode, "staged") == 0) {
            begin("cudaSetDevice"); RT(cudaSetDevice(0)); end("cudaSetDevice");
            begin("cudaFree0"); RT(cudaFree(nullptr)); end("cudaFree0");
            const size_t sizes[] = {1, 4096, 65536, 2U << 20, 16U << 20, (16U << 20) + 1};
            std::vector<void *> pointers;
            for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); ++i) {
                char name[64];
                std::snprintf(name, sizeof(name), "cudaMalloc_%zu", i);
                void *pointer = nullptr;
                begin(name, sizes[i]); RT(cudaMalloc(&pointer, sizes[i]));
                dprintf(2, "RETURN %s pointer=%p\n", name, pointer);
                end(name);
                pointers.push_back(pointer);
            }
            begin("cudaFree_all");
            for (void *pointer : pointers) RT(cudaFree(pointer));
            end("cudaFree_all");
            begin("cudaDeviceReset"); RT(cudaDeviceReset()); end("cudaDeviceReset");
        }
        else return 2;
    }
    dprintf(2, "RESULT PASS %s\n", mode);
    return 0;
}
