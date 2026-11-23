#include <cuda.h>
#include <cstdio>
#include <cstdlib>
static void check(CUresult r){if(r!=CUDA_SUCCESS){const char *name=nullptr;cuGetErrorName(r,&name);fprintf(stderr,"CUDA error %d: %s\n",int(r),name?name:"unknown");exit(1);}}
int main(){
 check(cuInit(0));CUdevice device;check(cuDeviceGet(&device,0));
 int supported=0;check(cuDeviceGetAttribute(&supported,CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,device));
 printf("VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED=%d\n",supported);
 CUcontext context;check(cuDevicePrimaryCtxRetain(&context,device));check(cuCtxSetCurrent(context));
 CUmemAllocationProp prop={};prop.type=CU_MEM_ALLOCATION_TYPE_PINNED;prop.location.type=CU_MEM_LOCATION_TYPE_DEVICE;prop.location.id=device;
 size_t gran=0;check(cuMemGetAllocationGranularity(&gran,&prop,CU_MEM_ALLOC_GRANULARITY_MINIMUM));
 size_t size=((16*1024*1024+gran-1)/gran)*gran;
 printf("minimum_granularity=%zu bytes\nallocation_size=%zu bytes\n",gran,size);
 CUmemGenericAllocationHandle handle;
 check(cuMemCreate(&handle,size,&prop,0));
 printf("cuMemCreate=CUDA_SUCCESS; no cuMemAddressReserve/cuMemMap called\n");
 check(cuMemRelease(handle));printf("cuMemRelease=CUDA_SUCCESS\n");
 check(cuCtxSetCurrent(nullptr));check(cuDevicePrimaryCtxRelease(device));return 0;
}
