#include <cuda_runtime_api.h>
#include <cuda.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <cstdio>
#include <cstdlib>
static void mark(const char* s) { write(2,s,strlen(s)); write(2,"\n",1); }
static void ck(cudaError_t r){if(r!=cudaSuccess){fprintf(stderr,"CUDA error: %s\n",cudaGetErrorString(r));exit(1);}}
static void dk(CUresult r){if(r!=CUDA_SUCCESS){fprintf(stderr,"Driver error: %d\n",int(r));exit(1);}}
int main(){
 const size_t n=16*1024*1024;
 ck(cudaSetDevice(0));ck(cudaFree(nullptr));
 cudaStream_t stream;ck(cudaStreamCreate(&stream));
 void* p=nullptr;
 mark("BEGIN host_mmap_and_touch");
 p=mmap(nullptr,n,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
 if(p==MAP_FAILED)return 2;
 memset(p,1,n);
 mark("END host_mmap_and_touch");
 munmap(p,n);
 mark("BEGIN cudaMalloc");ck(cudaMalloc(&p,n));mark("END cudaMalloc");ck(cudaFree(p));
 mark("BEGIN cudaMallocManaged");ck(cudaMallocManaged(&p,n));mark("END cudaMallocManaged");
 mark("BEGIN managed_CPU_first_touch");memset(p,2,n);mark("END managed_CPU_first_touch");ck(cudaFree(p));
 mark("BEGIN cudaMallocAsync");ck(cudaMallocAsync(&p,n,stream));ck(cudaStreamSynchronize(stream));mark("END cudaMallocAsync");
 ck(cudaFreeAsync(p,stream));ck(cudaStreamSynchronize(stream));
 CUmemAllocationProp prop={};prop.type=CU_MEM_ALLOCATION_TYPE_PINNED;prop.location.type=CU_MEM_LOCATION_TYPE_DEVICE;prop.location.id=0;
 size_t gran=0;dk(cuMemGetAllocationGranularity(&gran,&prop,CU_MEM_ALLOC_GRANULARITY_MINIMUM));
 size_t size=(n+gran-1)/gran*gran;CUmemGenericAllocationHandle h;CUdeviceptr addr;
 mark("BEGIN cuMemCreate");dk(cuMemCreate(&h,size,&prop,0));mark("END cuMemCreate");
 mark("BEGIN cuMemAddressReserve_and_Map");
 dk(cuMemAddressReserve(&addr,size,0,0,0));dk(cuMemMap(addr,size,0,h,0));
 CUmemAccessDesc access={};access.location=prop.location;access.flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
 dk(cuMemSetAccess(addr,size,&access,1));mark("END cuMemAddressReserve_and_Map");
 dk(cuMemUnmap(addr,size));dk(cuMemAddressFree(addr,size));dk(cuMemRelease(h));
 ck(cudaStreamDestroy(stream));
 mark("ALL TEST ALLOCATIONS RELEASED");
}
