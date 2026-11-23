#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include "nvos.h"

/* Observe original RM allocation inputs; forward every request unchanged. */
int ioctl(int fd, unsigned long request, ...)
{
    static int (*real_ioctl)(int, unsigned long, ...);
    if (!real_ioctl)
        real_ioctl = dlsym(RTLD_NEXT, "ioctl");

    va_list ap;
    va_start(ap, request);
    void *arg = va_arg(ap, void *);
    va_end(ap);

    int observed = 0;
    NvU32 observed_class = 0, observed_size = 0;
    NV_MEMORY_ALLOCATION_PARAMS memory = {0};
    if (arg && _IOC_TYPE(request) == 0x46 && _IOC_NR(request) == 0x2b) {
        NvU32 class_id = 0, params_size = 0;
        NvU64 params_address = 0;
        if (_IOC_SIZE(request) == sizeof(NVOS64_PARAMETERS)) {
            NVOS64_PARAMETERS header;
            memcpy(&header, arg, sizeof(header));
            class_id = header.hClass;
            params_size = header.paramsSize;
            memcpy(&params_address, &header.pAllocParms, sizeof(params_address));
        }
        else if (_IOC_SIZE(request) == sizeof(NVOS21_PARAMETERS)) {
            NVOS21_PARAMETERS header;
            memcpy(&header, arg, sizeof(header));
            class_id = header.hClass;
            params_size = header.paramsSize;
            memcpy(&params_address, &header.pAllocParms, sizeof(params_address));
        }
        observed_class = class_id;
        observed_size = params_size;
        if (class_id == 0x3e && params_address &&
            (params_size == 0 || params_size >= sizeof(memory))) {
            memcpy(&memory, (const void *)(uintptr_t)params_address, sizeof(memory));
            observed = 1;
        }
    }

    int result = real_ioctl(fd, request, arg);
    int saved_errno = errno;
    if (observed_class) {
        char buf[256];
        int n;
        if (observed)
            n = snprintf(buf, sizeof(buf),
                         "RM_SYSMEM_INPUT size=%llu attr2=0x%08x fixed_numa=%u numaNode=%d ioctl_result=%d\n",
                         (unsigned long long)memory.size, memory.attr2,
                         (memory.attr2 >> 7) & 1U, memory.numaNode, result);
        else
            n = snprintf(buf, sizeof(buf), "RM_ALLOC class=0x%x paramsSize=%u ioctl_result=%d\n",
                         observed_class, observed_size, result);
        if (n > 0 && (size_t)n < sizeof(buf)) {
            ssize_t written = write(STDERR_FILENO, buf, (size_t)n);
            (void)written;
        }
    }
    errno = saved_errno;
    return result;
}
