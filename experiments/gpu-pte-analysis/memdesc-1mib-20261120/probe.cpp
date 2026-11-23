#include <cuda_runtime_api.h>
#include <cstdio>
#include <cstdlib>
static void check(cudaError_t s) { if (s != cudaSuccess) { std::fprintf(stderr, "FAIL %s\n", cudaGetErrorString(s)); std::exit(1); } }
int main() {
    std::fprintf(stderr, "PHASE_BEGIN init\n");
    check(cudaSetDevice(0)); check(cudaFree(nullptr)); check(cudaDeviceSynchronize());
    std::fprintf(stderr, "PHASE_END init\n");
    void *pointers[3] = {};
    for (int i = 0; i < 3; ++i) {
        std::fprintf(stderr, "PHASE_BEGIN malloc_%d\n", i);
        check(cudaMalloc(&pointers[i], 1UL << 20));
        std::fprintf(stderr, "RETURN malloc_%d pointer=%p requested=1048576\n", i, pointers[i]);
        std::fprintf(stderr, "PHASE_END malloc_%d\n", i);
    }
    std::fprintf(stderr, "PHASE_BEGIN free\n");
    for (void *p : pointers) check(cudaFree(p));
    check(cudaDeviceReset());
    std::fprintf(stderr, "PHASE_END free\nRESULT PASS\n");
}
