#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "fault_monitor.h"

static void check(cudaError_t status)
{
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(status));
        std::exit(1);
    }
}

__global__ void warmup() {}
__global__ void write_one(int *data) { *data = 42; }

int main()
{
    check(cudaSetDevice(0));
    warmup<<<1, 1>>>();
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());

    GpuFaultMonitor monitor;
    constexpr size_t bytes = 4096;
    int *data = nullptr;
    check(cudaMallocManaged(&data, bytes));
    std::printf("Managed allocation: %p, %zu bytes\n", static_cast<void *>(data), bytes);

    write_one<<<1, 1>>>(data); // First data access: one GPU thread, one write.
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
    unsigned faults = monitor.report(data, bytes);

    int value = *data; // CPU first reads the data after GPU fault collection.
    check(cudaFree(data));
    bool pass = faults > 0 && value == 42;
    std::printf("GPU faults=%u, value=%d, %s\n", faults, value, pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
