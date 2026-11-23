#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

__device__ int static_data[1024];
__managed__ int static_managed[1024];

__global__ void use_heap(int *result)
{
    char *a = static_cast<char *>(malloc(4096));
    char *b = static_cast<char *>(__nv_aligned_device_malloc(4096, 64));
    int *c = new int[1024];
    int n = 0;
    if (a) { a[0] = 1; n += a[0]; free(a); }
    if (b) { b[0] = 1; n += b[0]; free(b); }
    if (c) { c[0] = 1; n += c[0]; delete[] c; }
    *result = n;
}

static void check(cudaError_t e, const char *s)
{
    if (e != cudaSuccess) {
        std::fprintf(stderr, "FAIL %s: %s\n", s, cudaGetErrorString(e));
        std::exit(1);
    }
}

int main()
{
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    std::fprintf(stderr, "BEGIN INIT\n");
    check(cudaSetDevice(0), "cudaSetDevice");
    check(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 8 * 1024 * 1024), "heap limit");
    check(cudaFree(nullptr), "initialize");
    int *result = nullptr;
    check(cudaMallocManaged(&result, sizeof(int)), "result allocation");
    std::fprintf(stderr, "END INIT\nBEGIN MODULE_GLOBAL\n");
    void *address = nullptr;
    check(cudaGetSymbolAddress(&address, static_data), "static device symbol");
    std::fprintf(stderr, "STATIC_DEVICE_POINTER %p\n", address);
    static_managed[0] = 123;
    std::fprintf(stderr, "STATIC_MANAGED_POINTER %p value=%d\n", static_managed, static_managed[0]);
    std::fprintf(stderr, "END MODULE_GLOBAL\nBEGIN DEVICE_HEAP\nSTEP first_kernel\n");
    use_heap<<<1, 1>>>(result);
    check(cudaGetLastError(), "launch 1");
    check(cudaDeviceSynchronize(), "synchronize 1");
    std::fprintf(stderr, "DEVICE_HEAP_RESULT first=%d\nSTEP second_kernel\n", *result);
    if (*result != 3) return 2;
    use_heap<<<1, 1>>>(result);
    check(cudaGetLastError(), "launch 2");
    check(cudaDeviceSynchronize(), "synchronize 2");
    std::fprintf(stderr, "DEVICE_HEAP_RESULT second=%d\nEND DEVICE_HEAP\n", *result);
    const bool ok = *result == 3;
    check(cudaFree(result), "free result");
    return ok ? 0 : 2;
}
