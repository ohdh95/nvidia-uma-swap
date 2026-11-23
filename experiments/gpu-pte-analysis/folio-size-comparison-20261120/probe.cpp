#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Stable entry points for the companion uprobes. No process memory is changed
// by the tracer. The markers distinguish allocation from materialization.
#define MARKER __attribute__((noinline, noclone, used, visibility("default")))
extern "C" MARKER void folio_probe_begin()
{ asm volatile("" ::: "memory"); }
extern "C" MARKER void folio_probe_allocated(void *p, size_t n)
{ asm volatile("" :: "r"(p), "r"(n) : "memory"); }
extern "C" MARKER void folio_probe_touched()
{ asm volatile("" ::: "memory"); }

static void cuda_check(cudaError_t e, const char *what)
{
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

static std::string quote(const std::string &s)
{
    std::ostringstream o;
    o << '"';
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') o << '\\' << c;
        else if (c == '\n') o << "\\n";
        else if (c == '\t') o << "\\t";
        else if (c < 32) o << "\\u" << std::hex << std::setw(4)
                           << std::setfill('0') << unsigned(c) << std::dec;
        else o << c;
    }
    o << '"';
    return o.str();
}

static uint64_t available_bytes()
{
    std::ifstream f("/proc/meminfo");
    std::string line;
    while (std::getline(f, line)) {
        unsigned long long kb;
        if (std::sscanf(line.c_str(), "MemAvailable: %llu kB", &kb) == 1)
            return kb * 1024;
    }
    throw std::runtime_error("Cannot read MemAvailable");
}

static void describe_range(void *p, size_t bytes, const std::string &mode,
                           size_t page_size, bool checked)
{
    uintptr_t start = reinterpret_cast<uintptr_t>(p), end = start + bytes;
    uintptr_t lo = start & ~(uintptr_t(page_size) - 1);
    size_t count = (end - lo + page_size - 1) / page_size;
    std::vector<uint64_t> entries(count);
    int pm = open("/proc/self/pagemap", O_RDONLY | O_CLOEXEC);
    if (pm < 0) throw std::runtime_error("Cannot open own pagemap");
    size_t done = 0, total = entries.size() * sizeof(uint64_t);
    while (done != total) {
        ssize_t n = pread(pm, reinterpret_cast<char *>(entries.data()) + done,
                          total - done, (lo / page_size) * 8 + done);
        if (n <= 0) { close(pm); throw std::runtime_error("pagemap read failed"); }
        done += size_t(n);
    }
    close(pm);
    size_t present = 0, swapped = 0, visible_pfns = 0;
    for (uint64_t e : entries) {
        if (e >> 63) {
            ++present;
            visible_pfns += bool(e & ((1ULL << 55) - 1));
        }
        swapped += (e >> 62) & 1;
    }
    int kflags = open("/proc/kpageflags", O_RDONLY | O_CLOEXEC);
    bool kflags_ok = kflags >= 0;
    if (kflags_ok) close(kflags);

    std::ostringstream address;
    address << "0x" << std::hex << start;
    std::cout << "PROBE_RESULT {\"mode\":" << quote(mode)
              << ",\"pid\":" << getpid() << ",\"uid\":" << getuid()
              << ",\"requested_bytes\":" << bytes
              << ",\"address\":" << quote(address.str())
              << ",\"base_page_bytes\":" << page_size
              << ",\"range_base_pages\":" << count
              << ",\"cpu_pte_present_pages\":" << present
              << ",\"cpu_pte_swapped_pages\":" << swapped
              << ",\"visible_pfn_pages\":" << visible_pfns
              << ",\"kpageflags_readable\":" << (kflags_ok ? "true" : "false")
              << ",\"data_check_pass\":" << (checked ? "true" : "false")
              << ",\"physical_folio_histogram\":null,\"vmas\":[";

    std::ifstream smaps("/proc/self/smaps");
    std::string line;
    bool in = false, first = true, first_metric = true;
    while (std::getline(smaps, line)) {
        unsigned long long a, b;
        if (std::sscanf(line.c_str(), "%llx-%llx", &a, &b) == 2) {
            if (in) std::cout << "]}";
            in = a < end && b > start;
            if (in) {
                if (!first) std::cout << ',';
                first = false;
                first_metric = true;
                std::cout << "{\"header\":" << quote(line) << ",\"metrics\":[";
            }
        } else if (in) {
            static const char *keys[] = {"KernelPageSize:", "MMUPageSize:",
                "Rss:", "Anonymous:", "AnonHugePages:", "Swap:",
                "THPeligible:", "VmFlags:"};
            for (const char *key : keys) {
                if (line.rfind(key, 0) == 0) {
                    if (!first_metric) std::cout << ',';
                    first_metric = false;
                    std::cout << quote(line);
                    break;
                }
            }
        }
    }
    if (in) std::cout << "]}";
    std::cout << "]}" << std::endl;
}

int main(int argc, char **argv)
{
    try {
        if (argc != 3) throw std::runtime_error("Usage: probe malloc|cudaMalloc MiB");
        std::string mode = argv[1];
        if (mode != "malloc" && mode != "cudaMalloc")
            throw std::runtime_error("mode must be malloc or cudaMalloc");
        size_t pos = 0;
        unsigned long long mib = std::stoull(argv[2], &pos);
        if (pos != std::strlen(argv[2]) || mib == 0 || mib > 4096)
            throw std::runtime_error("MiB must be in 1..4096");
        size_t bytes = size_t(mib) * 1024 * 1024;
        if (bytes > available_bytes() / 4)
            throw std::runtime_error("Test exceeds one quarter of MemAvailable");
        size_t page_size = size_t(sysconf(_SC_PAGESIZE));

        if (mode == "cudaMalloc") {
            cuda_check(cudaSetDevice(0), "cudaSetDevice");
            void *warm = nullptr;
            cuda_check(cudaMalloc(&warm, 1024 * 1024), "warm allocation");
            cuda_check(cudaMemset(warm, 0xa5, 1024 * 1024), "warm touch");
            cuda_check(cudaDeviceSynchronize(), "warm synchronization");
            cuda_check(cudaFree(warm), "warm free");
        }

        void *p = nullptr;
        folio_probe_begin();
        if (mode == "malloc") {
            p = std::malloc(bytes);
            if (!p) throw std::runtime_error("malloc failed");
        } else cuda_check(cudaMalloc(&p, bytes), "cudaMalloc");
        folio_probe_allocated(p, bytes);

        bool checked = true;
        if (mode == "malloc") {
            std::memset(p, 0xa5, bytes);
            volatile unsigned char *v = static_cast<volatile unsigned char *>(p);
            for (size_t i = 0; i < bytes; i += page_size)
                checked = checked && v[i] == 0xa5;
            checked = checked && v[bytes - 1] == 0xa5;
        } else {
            cuda_check(cudaMemset(p, 0xa5, bytes), "cudaMemset");
            cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
            unsigned char ends[2] = {};
            cuda_check(cudaMemcpy(&ends[0], p, 1, cudaMemcpyDeviceToHost), "check start");
            cuda_check(cudaMemcpy(&ends[1], static_cast<char *>(p) + bytes - 1,
                                  1, cudaMemcpyDeviceToHost), "check end");
            checked = ends[0] == 0xa5 && ends[1] == 0xa5;
        }
        folio_probe_touched();
        describe_range(p, bytes, mode, page_size, checked);
        if (mode == "malloc") std::free(p);
        else cuda_check(cudaFree(p), "cudaFree");
        return checked ? 0 : 2;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
