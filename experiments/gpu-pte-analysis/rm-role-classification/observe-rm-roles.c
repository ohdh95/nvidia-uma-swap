#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <execinfo.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include "nvos.h"
#include "alloc/alloc_channel.h"
#include "uvm_ioctl.h"

/* Read-only decoder. Forward original ioctl arguments without alteration. */
static void memory_json(char *buffer, size_t capacity, const NV_MEMORY_ALLOCATION_PARAMS *p)
{
    snprintf(buffer, capacity,
        "{\"owner\":%u,\"type\":%u,\"flags\":%u,\"width\":%u,\"height\":%u,\"pitch\":%d,"
        "\"attr\":%u,\"attr2\":%u,\"format\":%u,\"comprCovg\":%u,\"zcullCovg\":%u,"
        "\"rangeLo\":%llu,\"rangeHi\":%llu,\"size\":%llu,\"alignment\":%llu,\"offset\":%llu,"
        "\"limit\":%llu,\"address\":%llu,\"ctagOffset\":%u,\"hVASpace\":%u,"
        "\"internalflags\":%u,\"tag\":%u,\"numaNode\":%d}",
        p->owner, p->type, p->flags, p->width, p->height, p->pitch,
        p->attr, p->attr2, p->format, p->comprCovg, p->zcullCovg,
        (unsigned long long)p->rangeLo, (unsigned long long)p->rangeHi,
        (unsigned long long)p->size, (unsigned long long)p->alignment,
        (unsigned long long)p->offset, (unsigned long long)p->limit,
        (unsigned long long)(uintptr_t)p->address, p->ctagOffset, p->hVASpace,
        p->internalflags, p->tag, p->numaNode);
}

int ioctl(int fd, unsigned long request, ...)
{
    static int (*real_ioctl)(int, unsigned long, ...);
    if (!real_ioctl) real_ioctl = dlsym(RTLD_NEXT, "ioctl");
    va_list ap;
    va_start(ap, request);
    void *arg = va_arg(ap, void *);
    va_end(ap);
    int is_alloc = arg && _IOC_TYPE(request) == 0x46 && _IOC_NR(request) == 0x2b;
    NVOS64_PARAMETERS header = {0};
    if (is_alloc && _IOC_SIZE(request) == sizeof(NVOS64_PARAMETERS)) memcpy(&header, arg, sizeof(header));
    else if (is_alloc && _IOC_SIZE(request) == sizeof(NVOS21_PARAMETERS)) {
        NVOS21_PARAMETERS old;
        memcpy(&old, arg, sizeof(old));
        header.hRoot = old.hRoot; header.hObjectParent = old.hObjectParent;
        header.hObjectNew = old.hObjectNew; header.hClass = old.hClass;
        header.pAllocParms = old.pAllocParms; header.paramsSize = old.paramsSize;
    }
    else is_alloc = 0;
    NV_MEMORY_ALLOCATION_PARAMS before = {0};
    int is_memory = is_alloc && header.hClass == 0x3e && header.pAllocParms &&
        (!header.paramsSize || header.paramsSize >= sizeof(before));
    if (is_memory) memcpy(&before, (void *)(uintptr_t)header.pAllocParms, sizeof(before));
    // Decode only an RM channel allocation request, without inspecting CUDA APIs.
    NV_CHANNEL_ALLOC_PARAMS channel_before = {0};
    int is_channel = is_alloc && header.hClass == 0xc96f && header.pAllocParms;
    if (is_channel)
        memcpy(&channel_before, (void *)(uintptr_t)header.pAllocParms, sizeof(channel_before));
    int result = real_ioctl(fd, request, arg);
    int saved_errno = errno;
    char buffer[6000];
    int length = 0;
    long tid = syscall(SYS_gettid);
    if (is_alloc) {
        NvU32 status, object;
        if (_IOC_SIZE(request) == sizeof(NVOS64_PARAMETERS)) {
            NVOS64_PARAMETERS after; memcpy(&after, arg, sizeof(after));
            status = after.status; object = after.hObjectNew;
        }
        else {
            NVOS21_PARAMETERS after; memcpy(&after, arg, sizeof(after));
            status = after.status; object = after.hObjectNew;
        }
        if (is_memory) {
            NV_MEMORY_ALLOCATION_PARAMS after;
            char input[1800], output[1800];
            memcpy(&after, (void *)(uintptr_t)header.pAllocParms, sizeof(after));
            memory_json(input, sizeof(input), &before); memory_json(output, sizeof(output), &after);
            length = snprintf(buffer, sizeof(buffer),
                "OBS {\"event\":\"rm_memory\",\"tid\":%ld,\"fd\":%d,\"client\":%u,\"parent\":%u,"
                "\"object\":%u,\"class\":%u,\"paramsSize\":%u,\"rights\":%llu,\"result\":%d,"
                "\"status\":%u,\"in\":%s,\"out\":%s}\n",
                tid, fd, header.hRoot, header.hObjectParent, object, header.hClass, header.paramsSize,
                (unsigned long long)(uintptr_t)header.pRightsRequested, result, status, input, output);
        }
        else length = snprintf(buffer, sizeof(buffer),
            "OBS {\"event\":\"rm_object\",\"tid\":%ld,\"fd\":%d,\"client\":%u,\"parent\":%u,"
            "\"object\":%u,\"class\":%u,\"paramsSize\":%u,\"result\":%d,\"status\":%u}\n",
            tid, fd, header.hRoot, header.hObjectParent, object, header.hClass, header.paramsSize, result, status);
    }
    else if (arg && request == UVM_MAP_EXTERNAL_ALLOCATION) {
        const UVM_MAP_EXTERNAL_ALLOCATION_PARAMS *p = arg;
        length = snprintf(buffer, sizeof(buffer),
            "OBS {\"event\":\"uvm_map_external\",\"tid\":%ld,\"client\":%u,\"object\":%u,"
            "\"base\":%llu,\"length\":%llu,\"offset\":%llu,\"gpuCount\":%llu,"
            "\"mappingType\":%u,\"cachingType\":%u,\"formatType\":%u,\"elementBits\":%u,\"compressionType\":%u,"
            "\"result\":%d,\"status\":%u}\n",
            tid, p->hClient, p->hMemory, (unsigned long long)p->base, (unsigned long long)p->length,
            (unsigned long long)p->offset, (unsigned long long)p->gpuAttributesCount,
            p->perGpuAttributes[0].gpuMappingType, p->perGpuAttributes[0].gpuCachingType,
            p->perGpuAttributes[0].gpuFormatType, p->perGpuAttributes[0].gpuElementBits,
            p->perGpuAttributes[0].gpuCompressionType, result, p->rmStatus);
    }
    else if (arg && request == UVM_CREATE_EXTERNAL_RANGE) {
        const UVM_CREATE_EXTERNAL_RANGE_PARAMS *p = arg;
        length = snprintf(buffer, sizeof(buffer),
            "OBS {\"event\":\"uvm_create_external\",\"tid\":%ld,\"base\":%llu,\"length\":%llu,\"result\":%d,\"status\":%u}\n",
            tid, (unsigned long long)p->base, (unsigned long long)p->length, result, p->rmStatus);
    }
    else if (arg && _IOC_TYPE(request) == 0x46 && _IOC_NR(request) == 0x4e && _IOC_SIZE(request) >= sizeof(NVOS33_PARAMETERS)) {
        const NVOS33_PARAMETERS *p = arg;
        length = snprintf(buffer, sizeof(buffer),
            "OBS {\"event\":\"rm_map_cpu\",\"tid\":%ld,\"client\":%u,\"object\":%u,\"offset\":%llu,"
            "\"length\":%llu,\"flags\":%u,\"result\":%d,\"status\":%u}\n",
            tid, p->hClient, p->hMemory, (unsigned long long)p->offset, (unsigned long long)p->length,
            p->flags, result, p->status);
    }
    if (length > 0 && (size_t)length < sizeof(buffer)) {
        ssize_t ignored = write(2, buffer, length); (void)ignored;
    }
    if (is_channel) {
        NV_CHANNEL_ALLOC_PARAMS *p = &channel_before;
        int n = snprintf(buffer, sizeof(buffer),
            "OBS {\"event\":\"rm_channel\",\"client\":%u,\"object\":%u,\"gpfifo_va\":%llu,\"gpfifo_entries\":%u,\"vaspace\":%u,\"userd_handle\":%u,\"userd_offset\":%llu,\"error_handle\":%u,\"ecc_error_handle\":%u,\"engine_type\":%u,\"result\":%d}\n",
            header.hRoot, header.hObjectNew, (unsigned long long)p->gpFifoOffset,
            p->gpFifoEntries, p->hVASpace, p->hUserdMemory[0],
            (unsigned long long)p->userdOffset[0], p->hObjectError,
            p->hObjectEccError, p->engineType, result);
        if (n > 0 && (size_t)n < sizeof(buffer)) {
            ssize_t ignored = write(2, buffer, (size_t)n); (void)ignored;
        }
    }
    errno = saved_errno;
    return result;
}
