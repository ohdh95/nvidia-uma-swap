#pragma once

#include "uvm_ioctl.h"
#include <dirent.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <set>

// Subscribe only to the test process's existing UVM VA space. The driver
// pins these private queue/control buffers until the tools fd is closed.
class UvmEventMonitor {
    static constexpr unsigned queue_count = 65536;
    UvmEventEntry_V2 *queue = nullptr;
    UvmToolsEventControlData *control = nullptr;
    int tools_fd = -1;
    int uvm_fd = -1;
    FILE *events = nullptr;

public:
    bool start(const char *event_path)
    {
        queue = static_cast<UvmEventEntry_V2 *>(mmap(nullptr, queue_count * sizeof(*queue),
                    PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
        control = static_cast<UvmToolsEventControlData *>(mmap(nullptr, 4096,
                    PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
        if (queue == MAP_FAILED || control == MAP_FAILED)
            return false;
        std::memset(queue, 0, queue_count * sizeof(*queue));
        std::memset(control, 0, 4096);

        DIR *fds = opendir("/proc/self/fd");
        if (!fds)
            return false;
        while (dirent *entry = readdir(fds)) {
            char *end = nullptr;
            long candidate = std::strtol(entry->d_name, &end, 10);
            if (!*entry->d_name || *end || candidate < 0)
                continue;
            char path[128], target[256];
            std::snprintf(path, sizeof(path), "/proc/self/fd/%ld", candidate);
            ssize_t length = readlink(path, target, sizeof(target) - 1);
            if (length < 0)
                continue;
            target[length] = 0;
            if (std::strcmp(target, "/dev/nvidia-uvm") != 0)
                continue;
            int fd = open("/dev/nvidia-uvm-tools", O_RDWR | O_CLOEXEC);
            if (fd < 0)
                break;
            UVM_TOOLS_INIT_EVENT_TRACKER_V2_PARAMS params{};
            params.queueBuffer = reinterpret_cast<uintptr_t>(queue);
            params.queueBufferSize = queue_count; // Number of entries, not bytes.
            params.controlBuffer = reinterpret_cast<uintptr_t>(control);
            params.uvmFd = static_cast<NvU32>(candidate);
            int rc = ioctl(fd, UVM_TOOLS_INIT_EVENT_TRACKER_V2, &params);
            std::fprintf(stderr, "TRACKER candidate_fd=%ld rc=%d rmStatus=0x%x\n",
                         candidate, rc, params.rmStatus);
            if (rc == 0 && params.rmStatus == NV_OK) {
                tools_fd = fd;
                uvm_fd = static_cast<int>(candidate);
                break;
            }
            close(fd);
        }
        closedir(fds);
        if (tools_fd < 0)
            return false;

        UVM_TOOLS_EVENT_QUEUE_ENABLE_EVENTS_PARAMS enable{};
        enable.eventTypeFlags = UVM_EVENT_ENABLE_GPU_FAULT |
                                UVM_EVENT_ENABLE_GPU_FAULT_REPLAY |
                                UVM_EVENT_ENABLE_CPU_FAULT |
                                UVM_EVENT_ENABLE_FATAL_FAULT;
        int rc = ioctl(tools_fd, UVM_TOOLS_EVENT_QUEUE_ENABLE_EVENTS, &enable);
        std::fprintf(stderr, "ENABLE_EVENTS rc=%d rmStatus=0x%x entry_size=%zu\n",
                     rc, enable.rmStatus, sizeof(*queue));
        if (rc || enable.rmStatus != NV_OK)
            return false;
        events = std::fopen(event_path, "w");
        if (!events)
            return false;
        std::fprintf(events, "phase,type,address,in_allocation,fault_type,access_type,client_type,gpu_index,batch_id,cpu_timestamp,gpu_timestamp,fatal_reason\n");
        return true;
    }

    void drain(const char *phase, void *allocation, size_t bytes)
    {
        UVM_TOOLS_FLUSH_EVENTS_PARAMS flush{};
        int rc = ioctl(uvm_fd, UVM_TOOLS_FLUSH_EVENTS, &flush);
        if (rc || flush.rmStatus != NV_OK) {
            std::fprintf(stderr, "FLUSH rc=%d rmStatus=0x%x\n", rc, flush.rmStatus);
            std::exit(3);
        }
        uintptr_t base = reinterpret_cast<uintptr_t>(allocation);
        unsigned get = __atomic_load_n(&control->get_behind, __ATOMIC_SEQ_CST);
        unsigned put = __atomic_load_n(&control->put_behind, __ATOMIC_SEQ_CST);
        size_t gpu_faults = 0, target_faults = 0, cpu_faults = 0, replays = 0, fatal_faults = 0;
        std::set<NvU64> target_pages;
        while (get != put) {
            UvmEventEntry_V2 entry = queue[get];
            if (entry.eventData.eventType == UvmEventTypeGpuFault) {
                const auto &fault = entry.eventData.gpuFault;
                bool in_target = fault.address >= base && fault.address - base < bytes;
                ++gpu_faults;
                target_faults += in_target;
                if (in_target)
                    target_pages.insert(fault.address & ~NvU64(4095));
                std::fprintf(events, "%s,gpu_fault,0x%llx,%d,%u,%u,%u,%u,%u,%llu,%llu,\n",
                             phase, static_cast<unsigned long long>(fault.address), in_target,
                             fault.faultType, fault.accessType, fault.clientType, fault.gpuIndex,
                             fault.batchId, static_cast<unsigned long long>(fault.timeStamp),
                             static_cast<unsigned long long>(fault.timeStampGpu));
            }
            else if (entry.eventData.eventType == UvmEventTypeCpuFault) {
                const auto &fault = entry.eventData.cpuFault;
                bool in_target = fault.address >= base && fault.address - base < bytes;
                ++cpu_faults;
                std::fprintf(events, "%s,cpu_fault,0x%llx,%d,,%u,,,,%llu,,\n", phase,
                             static_cast<unsigned long long>(fault.address), in_target,
                             fault.accessType, static_cast<unsigned long long>(fault.timeStamp));
            }
            else if (entry.eventData.eventType == UvmEventTypeFatalFault) {
                const auto &fault = entry.eventData.fatalFault;
                bool in_target = fault.address >= base && fault.address - base < bytes;
                ++fatal_faults;
                std::fprintf(events, "%s,fatal_fault,0x%llx,%d,%u,%u,,%u,,%llu,,%u\n",
                             phase, static_cast<unsigned long long>(fault.address), in_target,
                             fault.faultType, fault.accessType, fault.processorIndex,
                             static_cast<unsigned long long>(fault.timeStamp), fault.reason);
            }
            else if (entry.eventData.eventType == UvmEventTypeGpuFaultReplay) {
                ++replays;
            }
            get = (get + 1) & (queue_count - 1);
        }
        __atomic_store_n(&control->get_ahead, get, __ATOMIC_SEQ_CST);
        __atomic_store_n(&control->get_behind, get, __ATOMIC_SEQ_CST);
        NvU64 dropped = 0;
        for (unsigned i = 0; i < UvmEventNumTypesAll; ++i)
            dropped += __atomic_load_n(&control->dropped[i], __ATOMIC_SEQ_CST);
        std::fprintf(stderr, "EVENTS phase=%s gpu_faults=%zu target_gpu_faults=%zu unique_target_4k_pages=%zu cpu_faults=%zu replays=%zu fatal_faults=%zu dropped_total=%llu\n",
                     phase, gpu_faults, target_faults, target_pages.size(), cpu_faults,
                     replays, fatal_faults, static_cast<unsigned long long>(dropped));
        std::fflush(events);
    }

    ~UvmEventMonitor()
    {
        if (tools_fd >= 0)
            close(tools_fd); // Unregister callbacks and unpin before munmap.
        if (events)
            std::fclose(events);
        if (queue && queue != MAP_FAILED)
            munmap(queue, queue_count * sizeof(*queue));
        if (control && control != MAP_FAILED)
            munmap(control, 4096);
    }
};
