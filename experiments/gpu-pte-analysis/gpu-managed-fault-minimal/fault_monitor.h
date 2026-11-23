#pragma once

#include "uvm_ioctl.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

// UVM tools plumbing, separate from the CUDA experiment.
// Uses the installed 580.173.02 driver's private UVM tools interface.
class GpuFaultMonitor {
    static constexpr unsigned capacity = 1024;
    UvmEventEntry_V2 *queue;
    UvmToolsEventControlData *control;
    int uvm_fd;
    int tools_fd;

    static void require(bool condition, const char *message)
    {
        if (!condition) {
            std::fprintf(stderr, "UVM monitor: %s\n", message);
            std::exit(2);
        }
    }

    template <typename Params>
    static void request(int fd, unsigned long command, Params &params)
    {
        int rc = ioctl(fd, command, &params);
        if (rc != 0 || params.rmStatus != NV_OK) {
            std::fprintf(stderr, "UVM ioctl 0x%lx: rc=%d, rmStatus=0x%x\n",
                         command, rc, params.rmStatus);
            std::exit(2);
        }
    }

    // Reuse CUDA's UVM fd to observe this process's actual VA space.
    static int find_uvm_fd()
    {
        DIR *fds = opendir("/proc/self/fd");
        require(fds != nullptr, "cannot inspect process file descriptors");
        int found = -1;
        while (dirent *entry = readdir(fds)) {
            char *end;
            long fd = std::strtol(entry->d_name, &end, 10);
            if (*end) continue;
            char path[128], target[256];
            std::snprintf(path, sizeof(path), "/proc/self/fd/%ld", fd);
            ssize_t size = readlink(path, target, sizeof(target) - 1);
            if (size < 0) continue;
            target[size] = '\0';
            if (std::strcmp(target, "/dev/nvidia-uvm") == 0) {
                found = static_cast<int>(fd);
                break;
            }
        }
        closedir(fds);
        require(found >= 0, "CUDA UVM fd not found");
        return found;
    }

public:
    GpuFaultMonitor()
    {
        uvm_fd = find_uvm_fd();
        tools_fd = open("/dev/nvidia-uvm-tools", O_RDWR | O_CLOEXEC);
        require(tools_fd >= 0, "cannot open /dev/nvidia-uvm-tools");
        queue = static_cast<UvmEventEntry_V2 *>(mmap(nullptr, capacity * sizeof(*queue),
                    PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
        control = static_cast<UvmToolsEventControlData *>(mmap(nullptr, 4096,
                    PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
        require(queue != MAP_FAILED && control != MAP_FAILED, "event buffer allocation failed");

        UVM_TOOLS_INIT_EVENT_TRACKER_V2_PARAMS init{};
        init.uvmFd = uvm_fd;
        init.queueBuffer = reinterpret_cast<uintptr_t>(queue);
        init.queueBufferSize = capacity; // Entry count, not byte count.
        init.controlBuffer = reinterpret_cast<uintptr_t>(control);
        request(tools_fd, UVM_TOOLS_INIT_EVENT_TRACKER_V2, init);

        UVM_TOOLS_EVENT_QUEUE_ENABLE_EVENTS_PARAMS enable{};
        enable.eventTypeFlags = UVM_EVENT_ENABLE_GPU_FAULT;
        request(tools_fd, UVM_TOOLS_EVENT_QUEUE_ENABLE_EVENTS, enable);
    }

    unsigned report(const void *data, size_t bytes)
    {
        UVM_TOOLS_FLUSH_EVENTS_PARAMS flush{};
        request(uvm_fd, UVM_TOOLS_FLUSH_EVENTS, flush);
        unsigned get = __atomic_load_n(&control->get_behind, __ATOMIC_SEQ_CST);
        unsigned put = __atomic_load_n(&control->put_behind, __ATOMIC_SEQ_CST);
        uintptr_t base = reinterpret_cast<uintptr_t>(data);
        unsigned count = 0;
        while (get != put) {
            const auto &event = queue[get].eventData;
            if (event.eventType == UvmEventTypeGpuFault) {
                const auto &fault = event.gpuFault;
                if (fault.address >= base && fault.address - base < bytes) {
                    const char *type = fault.faultType == UvmFaultTypeInvalidPte ? "InvalidPte" :
                                       fault.faultType == UvmFaultTypeInvalidPde ? "InvalidPde" : "other";
                    std::printf("GPU fault: eventType=%u, faultType=%u (%s), address=0x%llx\n",
                                 event.eventType, fault.faultType, type,
                                 static_cast<unsigned long long>(fault.address));
                    ++count;
                }
            }
            get = (get + 1) % capacity;
        }
        __atomic_store_n(&control->get_ahead, get, __ATOMIC_SEQ_CST);
        __atomic_store_n(&control->get_behind, get, __ATOMIC_SEQ_CST);
        NvU64 dropped = __atomic_load_n(&control->dropped[UvmEventTypeGpuFault], __ATOMIC_SEQ_CST);
        require(dropped == 0, "GPU fault events were dropped");
        return count;
    }

    GpuFaultMonitor(const GpuFaultMonitor &) = delete;
    GpuFaultMonitor &operator=(const GpuFaultMonitor &) = delete;

    ~GpuFaultMonitor()
    {
        close(tools_fd); // Let the driver unpin the buffers before munmap.
        munmap(queue, capacity * sizeof(*queue));
        munmap(control, 4096);
    }
};
