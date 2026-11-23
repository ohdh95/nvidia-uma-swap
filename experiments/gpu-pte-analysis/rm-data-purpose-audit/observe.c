/* Additional read-only RM observer; no CUDA API interception. */
#define ioctl dgx_role_base_ioctl
#include "../rm-role-classification/observe-rm-roles.c"
#undef ioctl
#include <sys/uio.h>

static size_t snapshot(void *out, const void *in, size_t length)
{
    struct iovec local = {out, length}, remote = {(void *)in, length};
    ssize_t n = process_vm_readv(getpid(), &local, 1, &remote, 1, 0);
    return n > 0 ? (size_t)n : 0;
}

int ioctl(int fd, unsigned long request, ...)
{
    va_list ap;
    va_start(ap, request);
    void *arg = va_arg(ap, void *);
    va_end(ap);
    NVOS54_PARAMETERS control = {0};
    unsigned char payload[512];
    size_t captured = 0;
    int is_control = arg && _IOC_TYPE(request) == 0x46 &&
        _IOC_NR(request) == 0x2a && _IOC_SIZE(request) >= sizeof(control);
    int is_legacy = arg && _IOC_TYPE(request) == 0x46 &&
        _IOC_NR(request) == 0x27 && _IOC_SIZE(request) >= sizeof(NVOS02_PARAMETERS);
    NVOS02_PARAMETERS legacy = {0};
    if (is_control && snapshot(&control, arg, sizeof(control)) == sizeof(control)) {
        size_t bytes = control.paramsSize < sizeof(payload) ? control.paramsSize : sizeof(payload);
        if (bytes && control.params)
            captured = snapshot(payload, (void *)(uintptr_t)control.params, bytes);
    }
    if (is_legacy) snapshot(&legacy, arg, sizeof(legacy));
    int result = dgx_role_base_ioctl(fd, request, arg);
    int saved_errno = errno;
    char output[1600];
    int length = 0;
    if (is_control) {
        NVOS54_PARAMETERS after = {0};
        snapshot(&after, arg, sizeof(after));
        char hex[2 * sizeof(payload) + 1];
        for (size_t i = 0; i < captured; ++i)
            snprintf(hex + 2 * i, 3, "%02x", payload[i]);
        hex[2 * captured] = 0;
        length = snprintf(output, sizeof(output),
            "OBS {\"event\":\"rm_control\",\"client\":%u,\"object\":%u,"
            "\"cmd\":%u,\"flags\":%u,\"paramsSize\":%u,\"captured\":%zu,"
            "\"input_hex\":\"%s\",\"result\":%d,\"status\":%u}\n",
            control.hClient, control.hObject, control.cmd, control.flags,
            control.paramsSize, captured, hex, result, after.status);
    }
    else if (is_legacy) {
        NVOS02_PARAMETERS after = {0};
        snapshot(&after, arg, sizeof(after));
        length = snprintf(output, sizeof(output),
            "OBS {\"event\":\"rm_alloc_legacy\",\"client\":%u,\"parent\":%u,"
            "\"object\":%u,\"class\":%u,\"flags\":%u,\"address_in\":%llu,"
            "\"address_out\":%llu,\"limit\":%llu,\"result\":%d,\"status\":%u}\n",
            after.hRoot, after.hObjectParent, after.hObjectNew, after.hClass,
            after.flags, (unsigned long long)(uintptr_t)legacy.pMemory,
            (unsigned long long)(uintptr_t)after.pMemory,
            (unsigned long long)after.limit, result, after.status);
    }
    if (length > 0 && (size_t)length < sizeof(output)) {
        ssize_t ignored = write(2, output, (size_t)length); (void)ignored;
    }
    errno = saved_errno;
    return result;
}
