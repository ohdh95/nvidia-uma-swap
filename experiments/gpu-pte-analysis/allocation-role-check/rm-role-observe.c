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
    NvU32 observed_client = 0, observed_object = 0;
    NV_MEMORY_ALLOCATION_PARAMS memory = {0};
    if (arg && _IOC_TYPE(request) == 0x46 && _IOC_NR(request) == 0x2b) {
        NvU32 class_id = 0, params_size = 0;
        NvU64 params_address = 0;
        if (_IOC_SIZE(request) == sizeof(NVOS64_PARAMETERS)) {
            NVOS64_PARAMETERS header;
            memcpy(&header, arg, sizeof(header));
            class_id = header.hClass;
            observed_client = header.hRoot;
            observed_object = header.hObjectNew;
            params_size = header.paramsSize;
            memcpy(&params_address, &header.pAllocParms, sizeof(params_address));
        }
        else if (_IOC_SIZE(request) == sizeof(NVOS21_PARAMETERS)) {
            NVOS21_PARAMETERS header;
            memcpy(&header, arg, sizeof(header));
            class_id = header.hClass;
            observed_client = header.hRoot;
            observed_object = header.hObjectNew;
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
        char buf[512];
        int n;
        if (observed)
            n = snprintf(buf, sizeof(buf),
                         "RM_SYSMEM client=0x%x object=0x%x owner=0x%x type=%u flags=0x%08x internal=0x%08x tag=0x%x attr=0x%08x attr2=0x%08x size=%llu ioctl_result=%d\n",
                         observed_client, observed_object, memory.owner, memory.type,
                         memory.flags, memory.internalflags, memory.tag, memory.attr,
                         memory.attr2, (unsigned long long)memory.size, result);
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
