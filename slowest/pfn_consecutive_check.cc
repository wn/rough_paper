#include <fcntl.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

constexpr int PAGE_SIZE = 4096;
constexpr int ELEMENT_COUNT = 1024 * 65536;
constexpr int ELEMENTS_PER_PAGE = PAGE_SIZE / sizeof(uint32_t);
constexpr int PAGE_COUNT = ELEMENT_COUNT / ELEMENTS_PER_PAGE;

bool physical_address(int fd, uintptr_t virtual_addr, uint64_t* physical_addr) {
    uint64_t entry = 0;
    off_t offset = (virtual_addr / PAGE_SIZE) * sizeof(entry);
    ssize_t bytes_read = pread(fd, &entry, sizeof(entry), offset);
    if (bytes_read != sizeof(entry)) {
        return false;
    }

    constexpr uint64_t present = 1ULL << 63;
    constexpr uint64_t pfn_mask = (1ULL << 55) - 1ULL;
    uint64_t pfn = entry & pfn_mask;
    if ((entry & present) == 0 || pfn == 0) {
        return false;
    }

    *physical_addr = pfn * PAGE_SIZE + (virtual_addr & (PAGE_SIZE - 1));
    return true;
}

int main() {
    uint32_t* data = nullptr;
    int error = posix_memalign(reinterpret_cast<void**>(&data), PAGE_SIZE,
                               ELEMENT_COUNT * sizeof(uint32_t));
    if (error != 0) {
        fprintf(stderr, "posix_memalign: %s\n", strerror(error));
        return 1;
    }

    for (int page = 0; page < PAGE_COUNT; ++page) {
        data[page * ELEMENTS_PER_PAGE] = static_cast<uint32_t>(page);
    }

    int fd = open("/proc/self/pagemap", O_RDONLY);
    if (fd < 0) {
        perror("open /proc/self/pagemap");
        free(data);
        return 1;
    }

    std::vector<uint64_t> pfns(PAGE_COUNT);
    uintptr_t base = reinterpret_cast<uintptr_t>(data);
    for (int page = 0; page < PAGE_COUNT; ++page) {
        uint64_t physical = 0;
        if (!physical_address(fd, base + static_cast<uintptr_t>(page) * PAGE_SIZE,
                              &physical)) {
            fprintf(stderr,
                    "could not read PFN for page %d; run with sudo or CAP_SYS_ADMIN\n",
                    page);
            close(fd);
            free(data);
            return 2;
        }
        pfns[page] = physical / PAGE_SIZE;
    }

    size_t plus_one = 0;
    size_t minus_one = 0;
    size_t other = 0;
    std::unordered_map<int64_t, size_t> delta_counts;
    for (int page = 1; page < PAGE_COUNT; ++page) {
        int64_t delta = static_cast<int64_t>(pfns[page]) -
                        static_cast<int64_t>(pfns[page - 1]);
        ++delta_counts[delta];
        if (delta == 1) {
            ++plus_one;
        } else if (delta == -1) {
            ++minus_one;
        } else {
            ++other;
        }
    }

    size_t longest_run = 1;
    size_t current_run = 1;
    for (int page = 1; page < PAGE_COUNT; ++page) {
        if (pfns[page] == pfns[page - 1] + 1) {
            ++current_run;
        } else {
            if (current_run > longest_run) {
                longest_run = current_run;
            }
            current_run = 1;
        }
    }
    if (current_run > longest_run) {
        longest_run = current_run;
    }

    printf("pages=%d\n", PAGE_COUNT);
    printf("first_pfn=%llu last_pfn=%llu\n",
           static_cast<unsigned long long>(pfns.front()),
           static_cast<unsigned long long>(pfns.back()));
    printf("adjacent_delta_plus_1=%zu\n", plus_one);
    printf("adjacent_delta_minus_1=%zu\n", minus_one);
    printf("adjacent_delta_other=%zu\n", other);
    printf("longest_plus_1_run_pages=%zu\n", longest_run);
    printf("top_deltas:\n");
    for (int printed = 0; printed < 8 && !delta_counts.empty(); ++printed) {
        auto best = delta_counts.begin();
        for (auto it = delta_counts.begin(); it != delta_counts.end(); ++it) {
            if (it->second > best->second) {
                best = it;
            }
        }
        printf("  delta=%lld count=%zu\n",
               static_cast<long long>(best->first), best->second);
        delta_counts.erase(best);
    }

    close(fd);
    free(data);
    return 0;
}
