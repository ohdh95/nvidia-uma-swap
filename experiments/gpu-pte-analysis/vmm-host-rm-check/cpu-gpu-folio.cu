#include <cuda.h>
#include <cuda_runtime.h>
#include <linux/kernel-page-flags.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <string>

#define DR(call) do { CUresult r=(call); if(r!=CUDA_SUCCESS){const char *s=nullptr; cuGetErrorName(r,&s); std::fprintf(stderr,"FAIL %s: %s\n",#call,s?s:"unknown"); std::exit(1);}} while(0)
#define RT(call) do { cudaError_t r=(call); if(r!=cudaSuccess){std::fprintf(stderr,"FAIL %s: %s\n",#call,cudaGetErrorString(r)); std::exit(1);}} while(0)

__global__ void increment_pages(int *p, size_t pages, size_t stride)
{
    size_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<pages) p[i*stride]+=1;
}

static void show_vma(CUdeviceptr address)
{
    std::ifstream smaps("/proc/self/smaps");
    std::string line;
    bool found=false;
    while(std::getline(smaps,line)) {
        unsigned long long lo,hi;
        if(std::sscanf(line.c_str(),"%llx-%llx",&lo,&hi)==2) {
            if(found) break;
            found=address>=lo && address<hi;
            if(found) std::printf("VMA %s\n",line.c_str());
        }
        if(found && (line.rfind("Rss:",0)==0 || line.rfind("Anonymous:",0)==0 ||
                     line.rfind("Locked:",0)==0 || line.rfind("Swap:",0)==0 ||
                     line.rfind("VmFlags:",0)==0)) std::printf("SMAPS %s\n",line.c_str());
    }
}

static void inspect_folios(CUdeviceptr address,size_t size,size_t page_size,const char *phase)
{
    int pm=open("/proc/self/pagemap",O_RDONLY|O_CLOEXEC);
    int flags=open("/proc/kpageflags",O_RDONLY|O_CLOEXEC);
    int flags_errno=flags<0?errno:0;
    if(pm<0) { std::printf("LRU_STATUS %s UNAVAILABLE pagemap: %s\n",phase,std::strerror(errno)); if(flags>=0)close(flags); return; }
    size_t present=0,pfn_nonzero=0,lru=0,unevictable=0,anon=0,swapbacked=0,head=0,tail=0,readable=0;
    for(size_t off=0;off<size;off+=page_size) {
        uint64_t entry=0,page_flags=0;
        if(pread(pm,&entry,8,((address+off)/page_size)*8)!=8) {
            std::printf("FAIL pagemap read\n"); std::exit(1);
        }
        if(!(entry&(1ULL<<63))) continue;
        ++present;
        uint64_t pfn=entry&((1ULL<<55)-1);
        if(pfn) ++pfn_nonzero;
        if(flags<0 || !pfn) continue;
        if(pread(flags,&page_flags,8,pfn*8)!=8) {
            std::printf("FAIL kpageflags read\n"); std::exit(1);
        }
        // This kernel's stable_page_flags() snapshots the owning folio,
        // so KPF_LRU also reflects the head folio for compound tail PFNs.
        ++readable;
        lru+=(page_flags>>KPF_LRU)&1;
        unevictable+=(page_flags>>KPF_UNEVICTABLE)&1;
        anon+=(page_flags>>KPF_ANON)&1;
        swapbacked+=(page_flags>>KPF_SWAPBACKED)&1;
        head+=(page_flags>>KPF_COMPOUND_HEAD)&1;
        tail+=(page_flags>>KPF_COMPOUND_TAIL)&1;
    }
    std::printf("PAGEMAP %s present=%zu/%zu nonzero_PFN=%zu\n",phase,present,size/page_size,pfn_nonzero);
    if(readable==size/page_size)
        std::printf("LRU_STATUS %s MEASURED base_pages=%zu LRU=%zu UNEVICTABLE=%zu ANON=%zu SWAPBACKED=%zu compound_heads=%zu compound_tails=%zu\n",phase,readable,lru,unevictable,anon,swapbacked,head,tail);
    else
        std::printf("LRU_STATUS %s UNAVAILABLE kpageflags=%s; PFN_zero_payload=%s; no LRU value was measured\n",phase,flags_errno?std::strerror(flags_errno):"readable",pfn_nonzero?"no":"yes (check CAP_SYS_ADMIN)");
    close(pm); if(flags>=0)close(flags);
}

int main()
{
    std::setvbuf(stdout,nullptr,_IONBF,0);
    std::setvbuf(stderr,nullptr,_IONBF,0);
    DR(cuInit(0)); RT(cudaSetDevice(0)); RT(cudaFree(nullptr));
    int *warm=nullptr; RT(cudaMalloc(&warm,sizeof(int))); RT(cudaMemset(warm,0,sizeof(int)));
    increment_pages<<<1,1>>>(warm,1,1); RT(cudaGetLastError()); RT(cudaDeviceSynchronize()); RT(cudaFree(warm));
    size_t page_size=static_cast<size_t>(sysconf(_SC_PAGESIZE));
    std::printf("PID=%d uid=%d page_size=%zu\n",getpid(),getuid(),page_size);
    for(CUmemLocationType type:{CU_MEM_LOCATION_TYPE_HOST,CU_MEM_LOCATION_TYPE_HOST_NUMA}) {
        const char *name=type==CU_MEM_LOCATION_TYPE_HOST?"HOST":"HOST_NUMA(0)";
        std::printf("BEGIN %s\n",name);
        CUmemAllocationProp prop{}; prop.type=CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type=type; prop.location.id=0;
        size_t gran=0; DR(cuMemGetAllocationGranularity(&gran,&prop,CU_MEM_ALLOC_GRANULARITY_MINIMUM));
        size_t size=((16*1024*1024+gran-1)/gran)*gran;
        CUmemGenericAllocationHandle handle; DR(cuMemCreate(&handle,size,&prop,0));
        CUdeviceptr address=0; DR(cuMemAddressReserve(&address,size,0,0,0)); DR(cuMemMap(address,size,0,handle,0));
        CUmemAccessDesc access[2]{}; access[0].location=prop.location;
        access[0].flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        access[1].location.type=CU_MEM_LOCATION_TYPE_DEVICE; access[1].location.id=0;
        access[1].flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        DR(cuMemSetAccess(address,size,access,2));
        std::printf("ALLOCATION %s address=0x%llx bytes=%zu\n",name,(unsigned long long)address,size);
        inspect_folios(address,size,page_size,"after_mapping");
        size_t pages=size/page_size,stride=page_size/sizeof(int);
        volatile int *cpu=reinterpret_cast<volatile int *>(address);
        size_t mismatches=0;
        for(size_t i=0;i<pages;++i) cpu[i*stride]=4096+static_cast<int>(i);
        for(size_t i=0;i<pages;++i) if(cpu[i*stride]!=4096+static_cast<int>(i))++mismatches;
        std::printf("CPU_READ_WRITE %s pages=%zu mismatches=%zu\n",name,pages,mismatches);
        if(mismatches)return 2;
        inspect_folios(address,size,page_size,"after_CPU_write");
        increment_pages<<<static_cast<unsigned>((pages+127)/128),128>>>(reinterpret_cast<int *>(address),pages,stride);
        RT(cudaGetLastError()); RT(cudaDeviceSynchronize());
        for(size_t i=0;i<pages;++i) if(cpu[i*stride]!=4097+static_cast<int>(i))++mismatches;
        std::printf("CPU_GPU_CPU %s pages=%zu mismatches=%zu %s\n",name,pages,mismatches,mismatches?"FAIL":"PASS");
        if(mismatches)return 2;
        show_vma(address);
        inspect_folios(address,size,page_size,"after_GPU_write");
        DR(cuMemUnmap(address,size)); DR(cuMemAddressFree(address,size)); DR(cuMemRelease(handle));
        std::printf("END %s\n",name);
    }
    RT(cudaDeviceReset());
    std::printf("MEMORY_TEST PASS (LRU results are reported separately)\n");
}
