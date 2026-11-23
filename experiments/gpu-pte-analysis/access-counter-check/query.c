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
#include "ctrl/ctrl2080/ctrl2080fb.h"
#include "ctrl/ctrl2080/ctrl2080mc.h"

/* Observe subdevice creation and issue only a read-only capability query. */
int ioctl(int fd, unsigned long request, ...)
{
    static int (*real_ioctl)(int, unsigned long, ...);
    if (!real_ioctl) real_ioctl = dlsym(RTLD_NEXT, "ioctl");
    va_list ap;
    va_start(ap, request);
    void *arg = va_arg(ap, void *);
    va_end(ap);

    int result = real_ioctl(fd, request, arg);
    int saved_errno = errno;
    NvU32 root = 0, object = 0, cls = 0, status = 1;
    if (result == 0 && arg && _IOC_TYPE(request) == 0x46 && _IOC_NR(request) == 0x2b) {
        if (_IOC_SIZE(request) == sizeof(NVOS64_PARAMETERS)) {
            NVOS64_PARAMETERS p;
            memcpy(&p, arg, sizeof(p));
            root = p.hRoot; object = p.hObjectNew; cls = p.hClass; status = p.status;
        }
        else if (_IOC_SIZE(request) == sizeof(NVOS21_PARAMETERS)) {
            NVOS21_PARAMETERS p;
            memcpy(&p, arg, sizeof(p));
            root = p.hRoot; object = p.hObjectNew; cls = p.hClass; status = p.status;
        }
    }
    if (cls == 0x2080 && status == 0) {
        NV2080_CTRL_FB_GET_INFO_V2_PARAMS info = {0};
        info.fbInfoListSize = 2;
        info.fbInfoList[0].index = NV2080_CTRL_FB_INFO_INDEX_IS_ZERO_FB;
        info.fbInfoList[1].index = NV2080_CTRL_FB_INFO_INDEX_ACCESS_COUNTER_BUFFER_COUNT;
        NVOS54_PARAMETERS control = {0};
        control.hClient = root;
        control.hObject = object;
        control.cmd = NV2080_CTRL_CMD_FB_GET_INFO_V2;
        control.paramsSize = sizeof(info);
        NvU64 pointer = (NvU64)(uintptr_t)&info;
        memcpy(&control.params, &pointer, sizeof(pointer));
        int query_result = real_ioctl(fd, _IOWR('F', 0x2a, NVOS54_PARAMETERS), &control);
        char line[320];
        int length = snprintf(line, sizeof(line),
            "CAPABILITY_QUERY ioctl_result=%d rm_status=0x%x IS_ZERO_FB=%u ACCESS_COUNTER_BUFFER_COUNT=%u client=0x%x subdevice=0x%x\n",
            query_result, control.status, info.fbInfoList[0].data, info.fbInfoList[1].data, root, object);
        if (length > 0 && (size_t)length < sizeof(line)) {
            ssize_t ignored = write(2, line, length); (void)ignored;
        }
        NV2080_CTRL_MC_GET_ARCH_INFO_PARAMS architecture = {0};
        control.cmd = NV2080_CTRL_CMD_MC_GET_ARCH_INFO;
        control.paramsSize = sizeof(architecture);
        control.status = 0;
        pointer = (NvU64)(uintptr_t)&architecture;
        memcpy(&control.params, &pointer, sizeof(pointer));
        query_result = real_ioctl(fd, _IOWR('F', 0x2a, NVOS54_PARAMETERS), &control);
        length = snprintf(line, sizeof(line),
            "ARCH_QUERY ioctl_result=%d rm_status=0x%x architecture=0x%x implementation=0x%x revision=0x%x\n",
            query_result, control.status, architecture.architecture, architecture.implementation, architecture.revision);
        if (length > 0 && (size_t)length < sizeof(line)) {
            ssize_t ignored = write(2, line, length); (void)ignored;
        }
    }
    errno = saved_errno;
    return result;
}
