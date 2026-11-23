#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/utsname.h>
#include <time.h>
#include <unistd.h>

static uint64_t pagemap_entry(int fd, const void *address, size_t page_size)
{
    uint64_t entry = 0;
    off_t offset = ((uintptr_t)address / page_size) * sizeof(entry);
    if (pread(fd, &entry, sizeof(entry), offset) != sizeof(entry)) {
        perror("pread pagemap");
        exit(1);
    }
    return entry;
}

static void show_entry(const char *stage, uint64_t entry)
{
    printf("PAGEMAP stage=%s raw=0x%016" PRIx64 " present=%u swapped=%u",
           stage, entry, (unsigned)(entry >> 63), (unsigned)((entry >> 62) & 1));
    if ((entry >> 62) & 1)
        printf(" swap_location=not_decoded");
    putchar('\n');
}

int main(void)
{
    struct utsname info;
    if (!uname(&info))
        printf("KERNEL %s\n", info.release);
    puts("NOTE: pagemap entries are not raw CPU PTEs; this kernel masks PFN/swap-location fields without CAP_SYS_ADMIN.");
    size_t page_size = (size_t)sysconf(_SC_PAGESIZE);
    size_t size = 16 * 1024 * 1024;
    size_t pages = size / page_size;
    unsigned char *data = mmap(NULL, size, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (data == MAP_FAILED) {
        perror("mmap");
        return 1;
    }
    if (madvise(data, size, MADV_NOHUGEPAGE)) {
        perror("MADV_NOHUGEPAGE");
        return 1;
    }
    int fd = open("/proc/self/pagemap", O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        perror("open pagemap");
        return 1;
    }
    printf("ALLOCATION address=%p bytes=%zu pages=%zu page_size=%zu\n",
           data, size, pages, page_size);
    show_entry("before_first_touch", pagemap_entry(fd, data, page_size));
    for (size_t i = 0; i < pages; ++i) {
        data[i * page_size] = (unsigned char)((i * 37 + 0x5a) % 251 + 1);
        data[(i + 1) * page_size - 1] = (unsigned char)((i * 53 + 0x35) % 251 + 1);
    }
    show_entry("after_write", pagemap_entry(fd, data, page_size));
    int rc = madvise(data, size, MADV_PAGEOUT);
    printf("MADV_PAGEOUT rc=%d errno=%d\n", rc, rc ? errno : 0);
    if (rc) {
        perror("MADV_PAGEOUT");
        return 1;
    }

    // Examine page-table metadata without dereferencing the test allocation.
    // PAGEOUT is a reclaim hint; only a confirmed swapped entry counts.
    size_t swapped_pages = 0, target_index = pages;
    struct timespec delay = {.tv_sec = 0, .tv_nsec = 20 * 1000 * 1000};
    for (unsigned attempt = 0; attempt < 100; ++attempt) {
        swapped_pages = 0;
        target_index = pages;
        for (size_t i = 0; i < pages; ++i) {
            uint64_t entry = pagemap_entry(fd, data + i * page_size, page_size);
            if ((entry >> 62) & 1) {
                ++swapped_pages;
                if (target_index == pages)
                    target_index = i;
            }
        }
        if (swapped_pages)
            break;
        nanosleep(&delay, NULL);
    }
    printf("AFTER_PAGEOUT swapped_pages=%zu total_pages=%zu\n", swapped_pages, pages);
    if (target_index == pages) {
        fprintf(stderr, "INCONCLUSIVE: PAGEOUT did not produce a swapped entry.\n");
        return 2;
    }
    volatile unsigned char *target = data + target_index * page_size;
    uint64_t before = pagemap_entry(fd, (const void *)target, page_size);
    show_entry("before_swapin_load", before);

    struct rusage warm, usage_before, usage_after;
    getrusage(RUSAGE_THREAD, &warm);
    getrusage(RUSAGE_THREAD, &usage_before);
    unsigned char value = *target;
    getrusage(RUSAGE_THREAD, &usage_after);
    long minor = usage_after.ru_minflt - usage_before.ru_minflt;
    long major = usage_after.ru_majflt - usage_before.ru_majflt;
    printf("ONE_VOLATILE_LOAD target_page=%zu value=%u minor_fault_delta=%ld major_fault_delta=%ld\n",
           target_index, value, minor, major);
    uint64_t after = pagemap_entry(fd, (const void *)target, page_size);
    show_entry("after_swapin_load", after);

    size_t mismatches = 0;
    for (size_t i = 0; i < pages; ++i) {
        if (data[i * page_size] != (unsigned char)((i * 37 + 0x5a) % 251 + 1))
            ++mismatches;
        if (data[(i + 1) * page_size - 1] != (unsigned char)((i * 53 + 0x35) % 251 + 1))
            ++mismatches;
    }
    printf("VERIFY locations=%zu mismatches=%zu\n", pages * 2, mismatches);
    int success = ((before >> 62) & 1) && !(before >> 63) &&
                  (after >> 63) && !((after >> 62) & 1) &&
                  minor + major > 0 && !mismatches;
    printf("RESULT %s\n", success ? "PASS" : "FAIL");
    close(fd);
    munmap(data, size);
    return success ? 0 : 3;
}
