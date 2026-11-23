#include <cuda_runtime_api.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void mark(const char *text)
{
    write(STDERR_FILENO, text, strlen(text));
    write(STDERR_FILENO, "\n", 1);
}

static void check(cudaError_t result)
{
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(result));
        exit(EXIT_FAILURE);
    }
}

int main()
{
    mark("BEGIN CUDA_INIT");
    check(cudaSetDevice(0));
    check(cudaFree(nullptr));
    mark("END CUDA_INIT");

    void *ptr = nullptr;
    mark("BEGIN CUDA_MALLOC_ONLY");
    cudaError_t result = cudaMalloc(&ptr, 16 * 1024 * 1024);
    mark("END CUDA_MALLOC_ONLY");
    check(result);

    // No CPU/GPU access to the allocation, and no managed allocation.
    mark("BEGIN CUDA_FREE");
    check(cudaFree(ptr));
    mark("END CUDA_FREE");
    return EXIT_SUCCESS;
}
