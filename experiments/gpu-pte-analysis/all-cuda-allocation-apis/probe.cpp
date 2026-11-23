#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <string>
#include <sys/mman.h>

constexpr size_t bytes = 16 * 1024 * 1024;
static cudaExtent make_extent(size_t width, size_t height, size_t depth) {
    return cudaExtent{width, height, depth};
}
struct Failure { const char *operation; int code; const char *message; };
static void rt(cudaError_t e, const char *name) {
    if (e != cudaSuccess) throw Failure{name, int(e), cudaGetErrorString(e)};
}
static void dr(CUresult e, const char *name) {
    if (e != CUDA_SUCCESS) {
        const char *s = nullptr;
        cuGetErrorString(e, &s);
        throw Failure{name, int(e), s ? s : "unknown"};
    }
}
#define RT(x) rt((x), #x)
#define DR(x) dr((x), #x)
static void step(const char *s) { std::fprintf(stderr, "STEP %s\n", s); }
static void touch(void *p) {
    step("CPU_first_touch");
    auto q = static_cast<volatile unsigned char *>(p);
    for (size_t i = 0; i < bytes; i += 4096) q[i] = 0x5a;
    q[bytes - 1] = 0xa5;
}
static void smaps(void *p) {
    std::ifstream f("/proc/self/smaps");
    std::string s;
    bool found = false;
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    std::fprintf(stderr, "POINTER %p\n", p);
    while (std::getline(f, s)) {
        unsigned long long lo, hi; char prot[5];
        if (std::sscanf(s.c_str(), "%llx-%llx %4s", &lo, &hi, prot) == 3) {
            if (found) break;
            found = a >= lo && a < hi;
            if (found) std::fprintf(stderr, "VMA %s\n", s.c_str());
        }
        if (found && (s.rfind("Rss:", 0) == 0 || s.rfind("Anonymous:", 0) == 0 ||
                      s.rfind("Locked:", 0) == 0 || s.rfind("VmFlags:", 0) == 0))
            std::fprintf(stderr, "SMAPS %s\n", s.c_str());
    }
}

static void run(const std::string &name) {
    if (name == "cudaMalloc") {
        void *p = nullptr; RT(cudaMalloc(&p, bytes)); smaps(p);
        step("cudaFree"); RT(cudaFree(p));
    } else if (name == "cudaMallocPitch") {
        void *p = nullptr; size_t pitch;
        RT(cudaMallocPitch(&p, &pitch, 4096, 4096)); smaps(p);
        step("cudaFree"); RT(cudaFree(p));
    } else if (name == "cudaMalloc3D") {
        cudaPitchedPtr p{};
        RT(cudaMalloc3D(&p, make_extent(256, 256, 256))); smaps(p.ptr);
        step("cudaFree"); RT(cudaFree(p.ptr));
    } else if (name == "cudaMallocManaged" || name == "cudaMallocManagedHost") {
        void *p = nullptr;
        RT(cudaMallocManaged(&p, bytes, name == "cudaMallocManaged" ? cudaMemAttachGlobal : cudaMemAttachHost));
        touch(p); smaps(p); step("cudaFree"); RT(cudaFree(p));
    } else if (name == "cudaMallocHost" || name.rfind("cudaHostAlloc", 0) == 0) {
        void *p = nullptr;
        if (name == "cudaMallocHost") RT(cudaMallocHost(&p, bytes));
        else {
            unsigned flags = name == "cudaHostAllocMapped" ? cudaHostAllocMapped :
                             name == "cudaHostAllocWC" ? cudaHostAllocWriteCombined : cudaHostAllocDefault;
            RT(cudaHostAlloc(&p, bytes, flags));
        }
        touch(p); smaps(p); step("cudaFreeHost"); RT(cudaFreeHost(p));
    } else if (name == "cudaHostRegister" || name == "cuMemHostRegister") {
        step("anonymous_mmap");
        void *p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) throw Failure{"mmap", -1, "failed"};
        touch(p); smaps(p); step("register");
        if (name == "cudaHostRegister") RT(cudaHostRegister(p, bytes, cudaHostRegisterDefault));
        else DR(cuMemHostRegister(p, bytes, 0));
        smaps(p); step("unregister");
        if (name == "cudaHostRegister") RT(cudaHostUnregister(p));
        else DR(cuMemHostUnregister(p));
        munmap(p, bytes);
    } else if (name == "cudaMallocArray" || name == "cudaMallocArraySparse" || name == "cudaMalloc3DArray") {
        cudaArray_t a = nullptr;
        cudaChannelFormatDesc d = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        if (name == "cudaMalloc3DArray") RT(cudaMalloc3DArray(&a, &d, make_extent(128, 128, 128), 0));
        else RT(cudaMallocArray(&a, &d, 1024, 1024, name == "cudaMallocArraySparse" ? cudaArraySparse : 0));
        step("cudaFreeArray"); RT(cudaFreeArray(a));
    } else if (name == "cudaMallocMipmappedArray" || name == "cudaMallocMipmappedArraySparse") {
        cudaMipmappedArray_t a = nullptr;
        cudaChannelFormatDesc d = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        RT(cudaMallocMipmappedArray(&a, &d, make_extent(1024, 1024, 0), 4,
                                    name == "cudaMallocMipmappedArraySparse" ? cudaArraySparse : 0));
        step("cudaFreeMipmappedArray"); RT(cudaFreeMipmappedArray(a));
    } else if (name == "cudaMallocAsync") {
        cudaStream_t s; RT(cudaStreamCreate(&s));
        void *p = nullptr; step("cudaMallocAsync"); RT(cudaMallocAsync(&p, bytes, s));
        RT(cudaStreamSynchronize(s)); smaps(p);
        step("cudaFreeAsync"); RT(cudaFreeAsync(p, s)); RT(cudaStreamSynchronize(s));
        RT(cudaStreamDestroy(s));
    } else if (name.rfind("cudaPool", 0) == 0) {
        cudaMemPoolProps props{};
        props.allocType = name == "cudaPoolManaged" ? cudaMemAllocationTypeManaged : cudaMemAllocationTypePinned;
        props.location.type = name == "cudaPoolHost" ? cudaMemLocationTypeHost :
                              name == "cudaPoolHostNuma" ? cudaMemLocationTypeHostNuma :
                              name == "cudaPoolManaged" ? cudaMemLocationTypeNone : cudaMemLocationTypeDevice;
        props.location.id = 0;
        cudaMemPool_t pool; step("cudaMemPoolCreate"); RT(cudaMemPoolCreate(&pool, &props));
        cudaStream_t s; RT(cudaStreamCreate(&s));
        void *p = nullptr; step("cudaMallocFromPoolAsync"); RT(cudaMallocFromPoolAsync(&p, bytes, pool, s));
        RT(cudaStreamSynchronize(s));
        if (name != "cudaPoolDevice") touch(p);
        smaps(p);
        step("cudaFreeAsync"); RT(cudaFreeAsync(p, s)); RT(cudaStreamSynchronize(s));
        step("cudaMemPoolTrimTo"); RT(cudaMemPoolTrimTo(pool, 0));
        RT(cudaMemPoolDestroy(pool)); RT(cudaStreamDestroy(s));
    } else if (name == "cuMemAlloc" || name == "cuMemAllocPitch") {
        CUdeviceptr p = 0; size_t pitch = 0;
        if (name == "cuMemAlloc") DR(cuMemAlloc(&p, bytes));
        else DR(cuMemAllocPitch(&p, &pitch, 4096, 4096, 4));
        smaps(reinterpret_cast<void *>(p)); step("cuMemFree"); DR(cuMemFree(p));
    } else if (name == "cuMemAllocManaged") {
        CUdeviceptr p = 0; DR(cuMemAllocManaged(&p, bytes, CU_MEM_ATTACH_GLOBAL));
        touch(reinterpret_cast<void *>(p)); smaps(reinterpret_cast<void *>(p));
        step("cuMemFree"); DR(cuMemFree(p));
    } else if (name == "cuMemAllocHost" || name == "cuMemHostAlloc") {
        void *p = nullptr;
        if (name == "cuMemAllocHost") DR(cuMemAllocHost(&p, bytes));
        else DR(cuMemHostAlloc(&p, bytes, 0));
        touch(p); smaps(p); step("cuMemFreeHost"); DR(cuMemFreeHost(p));
    } else if (name == "cuArrayCreate" || name == "cuArray3DCreate" || name == "cuArray3DCreateSparse") {
        CUarray a = nullptr;
        if (name == "cuArrayCreate") {
            CUDA_ARRAY_DESCRIPTOR d{}; d.Width=1024; d.Height=1024; d.Format=CU_AD_FORMAT_FLOAT; d.NumChannels=1;
            DR(cuArrayCreate(&a, &d));
        } else {
            CUDA_ARRAY3D_DESCRIPTOR d{}; d.Width=128; d.Height=128; d.Depth=128; d.Format=CU_AD_FORMAT_FLOAT; d.NumChannels=1;
            d.Flags = name == "cuArray3DCreateSparse" ? CUDA_ARRAY3D_SPARSE : 0;
            DR(cuArray3DCreate(&a, &d));
        }
        step("cuArrayDestroy"); DR(cuArrayDestroy(a));
    } else if (name == "cuMipmappedArrayCreate") {
        CUmipmappedArray a = nullptr;
        CUDA_ARRAY3D_DESCRIPTOR d{}; d.Width=1024; d.Height=1024; d.Format=CU_AD_FORMAT_FLOAT; d.NumChannels=1;
        DR(cuMipmappedArrayCreate(&a, &d, 4)); step("cuMipmappedArrayDestroy"); DR(cuMipmappedArrayDestroy(a));
    } else if (name.rfind("cuMemCreate", 0) == 0) {
        CUmemAllocationProp props{}; props.type=CU_MEM_ALLOCATION_TYPE_PINNED;
        props.location.type = name == "cuMemCreateHost" ? CU_MEM_LOCATION_TYPE_HOST :
                              name == "cuMemCreateHostNuma" ? CU_MEM_LOCATION_TYPE_HOST_NUMA : CU_MEM_LOCATION_TYPE_DEVICE;
        props.location.id=0;
        if (name == "cuMemCreateTilePool") props.allocFlags.usage=CU_MEM_CREATE_USAGE_TILE_POOL;
        size_t gran=0; DR(cuMemGetAllocationGranularity(&gran, &props, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
        size_t n = (bytes + gran - 1) / gran * gran;
        CUmemGenericAllocationHandle h; step("cuMemCreate"); DR(cuMemCreate(&h, n, &props, 0));
        step("cuMemRelease"); DR(cuMemRelease(h));
    } else if (name == "cuMemAllocAsync") {
        CUstream s; DR(cuStreamCreate(&s, CU_STREAM_DEFAULT)); CUdeviceptr p;
        step("cuMemAllocAsync"); DR(cuMemAllocAsync(&p, bytes, s)); DR(cuStreamSynchronize(s));
        smaps(reinterpret_cast<void *>(p)); step("cuMemFreeAsync"); DR(cuMemFreeAsync(p, s));
        DR(cuStreamSynchronize(s)); DR(cuStreamDestroy(s));
    } else if (name.rfind("cuPool", 0) == 0) {
        CUmemPoolProps props{}; props.allocType = name == "cuPoolManaged" ? CU_MEM_ALLOCATION_TYPE_MANAGED : CU_MEM_ALLOCATION_TYPE_PINNED;
        props.location.type = name == "cuPoolHost" ? CU_MEM_LOCATION_TYPE_HOST :
                              name == "cuPoolHostNuma" ? CU_MEM_LOCATION_TYPE_HOST_NUMA :
                              name == "cuPoolManaged" ? CU_MEM_LOCATION_TYPE_NONE : CU_MEM_LOCATION_TYPE_DEVICE;
        props.location.id=0;
        CUmemoryPool pool; step("cuMemPoolCreate"); DR(cuMemPoolCreate(&pool, &props));
        CUstream s; DR(cuStreamCreate(&s, CU_STREAM_DEFAULT)); CUdeviceptr p;
        step("cuMemAllocFromPoolAsync"); DR(cuMemAllocFromPoolAsync(&p, bytes, pool, s)); DR(cuStreamSynchronize(s));
        if (name != "cuPoolDevice") touch(reinterpret_cast<void *>(p));
        smaps(reinterpret_cast<void *>(p)); step("cuMemFreeAsync"); DR(cuMemFreeAsync(p, s));
        DR(cuStreamSynchronize(s)); step("cuMemPoolTrimTo"); DR(cuMemPoolTrimTo(pool, 0));
        DR(cuMemPoolDestroy(pool)); DR(cuStreamDestroy(s));
    } else if (name == "cudaGraphMemory") {
        cudaGraph_t graph; cudaGraphExec_t exec; cudaGraphNode_t alloc, free;
        RT(cudaGraphCreate(&graph, 0));
        cudaMemAllocNodeParams p{}; p.poolProps.allocType=cudaMemAllocationTypePinned;
        p.poolProps.location.type=cudaMemLocationTypeDevice; p.poolProps.location.id=0; p.bytesize=bytes;
        step("cudaGraphAddMemAllocNode"); RT(cudaGraphAddMemAllocNode(&alloc, graph, nullptr, 0, &p));
        RT(cudaGraphAddMemFreeNode(&free, graph, &alloc, 1, p.dptr));
        step("cudaGraphInstantiate"); RT(cudaGraphInstantiate(&exec, graph, 0));
        step("cudaGraphLaunch"); RT(cudaGraphLaunch(exec, nullptr)); RT(cudaDeviceSynchronize());
        RT(cudaGraphExecDestroy(exec)); RT(cudaGraphDestroy(graph));
        step("cudaDeviceGraphMemTrim"); RT(cudaDeviceGraphMemTrim(0));
    } else if (name == "cuGraphMemory") {
        CUgraph graph; CUgraphExec exec; CUgraphNode alloc, free;
        DR(cuGraphCreate(&graph, 0));
        CUDA_MEM_ALLOC_NODE_PARAMS p{}; p.poolProps.allocType=CU_MEM_ALLOCATION_TYPE_PINNED;
        p.poolProps.location.type=CU_MEM_LOCATION_TYPE_DEVICE; p.poolProps.location.id=0; p.bytesize=bytes;
        step("cuGraphAddMemAllocNode"); DR(cuGraphAddMemAllocNode(&alloc, graph, nullptr, 0, &p));
        DR(cuGraphAddMemFreeNode(&free, graph, &alloc, 1, p.dptr));
        step("cuGraphInstantiate"); DR(cuGraphInstantiate(&exec, graph, 0));
        step("cuGraphLaunch"); DR(cuGraphLaunch(exec, nullptr)); DR(cuCtxSynchronize());
        DR(cuGraphExecDestroy(exec)); DR(cuGraphDestroy(graph));
        step("cuDeviceGraphMemTrim"); DR(cuDeviceGraphMemTrim(0));
    } else throw Failure{"test selection", -1, "unknown case"};
}

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    std::fprintf(stderr, "BEGIN INIT\n");
    try {
        RT(cudaSetDevice(0)); RT(cudaFree(nullptr)); DR(cuInit(0));
        std::fprintf(stderr, "END INIT\nBEGIN API %s\n", argv[1]);
        run(argv[1]);
        std::fprintf(stderr, "RESULT PASS %s\nEND API %s\n", argv[1], argv[1]);
        return 0;
    } catch (const Failure &e) {
        std::fprintf(stderr, "RESULT FAIL %s operation=%s code=%d message=%s\nEND API %s\n",
                     argv[1], e.operation, e.code, e.message, argv[1]);
        return 1;
    }
}
