#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

constexpr int PAGE_SIZE = 4096;
constexpr int CACHELINE_SIZE = 64;
constexpr int ELEMENT_COUNT = 1024 * 65536;  // 65M
constexpr int ALLOCATION_ALIGNMENT = PAGE_SIZE;

#ifndef STRIDE
#define STRIDE 8
#endif

#ifndef DRAM_ROW_SHIFT
#define DRAM_ROW_SHIFT 18
#endif

#ifndef DRAM_BANK_MASK
#define DRAM_BANK_MASK \
    ((((1ULL << DRAM_ROW_SHIFT) - 1ULL) & ~((1ULL << 12) - 1ULL)))
#endif

#ifndef DRAM_BANK_XOR_SHIFT
#define DRAM_BANK_XOR_SHIFT 0
#endif

static_assert(DRAM_ROW_SHIFT > 12);

uint64_t rdtsc_start() {
    uint32_t lo;
    uint32_t hi;
    __asm__ __volatile__(
        "lfence\n\t"
        "rdtsc"
        : "=a"(lo), "=d"(hi)
        :
        : "memory");
    return ((uint64_t)hi << 32) | lo;
}

uint64_t rdtsc_end() {
    uint32_t lo;
    uint32_t hi;
    __asm__ __volatile__(
        "rdtscp\n\t"
        "lfence"
        : "=a"(lo), "=d"(hi)
        :
        : "rcx", "memory");
    return ((uint64_t)hi << 32) | lo;
}

void print_cycles(char const* label, uint64_t cycles) {
    printf("%-46s %12lu\n", label, cycles);
}

static inline void do_not_optimize(uint32_t value) {
    __asm__ __volatile__("" : : "r,m"(value) : "memory");
}

void fill_data(uint32_t* data) {
    for (uint32_t i = 0; i < ELEMENT_COUNT; ++i) {
        data[i] = rand();
    }
}

// overflow is expected but we don't really care about the actual sum, do we.
uint32_t accumulator(uint32_t const* data, uint32_t const* positions) {
    uint32_t total = 0;
    for (uint32_t i = 0; i < ELEMENT_COUNT; ++i) {
        uint32_t pos = positions[i];
        total += data[pos];
    }
    return total;
}

void linear(uint32_t const*, uint32_t* positions) {
    for (uint32_t i = 0; i < ELEMENT_COUNT; ++i) {
        positions[i] = i;
    }
}

void fisher_yates_shuffle(uint32_t const* data, uint32_t* positions) {
    linear(data, positions);
    uint32_t remaining = ELEMENT_COUNT;
    for (uint32_t i = 0; i < ELEMENT_COUNT; ++i) {
        uint32_t random = rand() % remaining;
        uint32_t tmp = positions[i];
        positions[i] = positions[i + random];
        positions[i + random] = tmp;
        --remaining;
    }
}

void separated_by_a_cacheline(uint32_t const* data, uint32_t* positions) {
    linear(data, positions);
    constexpr int element_ELEMENT_COUNT_per_cacheline =
        CACHELINE_SIZE / sizeof(uint32_t);
    constexpr int cacheline_ELEMENT_COUNT =
        ELEMENT_COUNT / element_ELEMENT_COUNT_per_cacheline;
    static_assert(ELEMENT_COUNT % element_ELEMENT_COUNT_per_cacheline == 0);
    int current = 0;
    for (int element_index = 0;
         element_index < element_ELEMENT_COUNT_per_cacheline; ++element_index) {
        for (int cacheline_index = 0; cacheline_index < cacheline_ELEMENT_COUNT;
             ++cacheline_index) {
            positions[current] =
                cacheline_index * element_ELEMENT_COUNT_per_cacheline +
                element_index;
            ++current;
        }
    }
}

void separated_by_a_page(uint32_t const* data, uint32_t* positions) {
    linear(data, positions);
    constexpr int element_ELEMENT_COUNT_per_page = PAGE_SIZE / sizeof(uint32_t);
    constexpr int page_ELEMENT_COUNT =
        ELEMENT_COUNT / element_ELEMENT_COUNT_per_page;
    static_assert(ELEMENT_COUNT % element_ELEMENT_COUNT_per_page == 0);
    int current = 0;
    for (int element_index = 0; element_index < element_ELEMENT_COUNT_per_page;
         ++element_index) {
        for (int page_index = 0; page_index < page_ELEMENT_COUNT;
             ++page_index) {
            positions[current] =
                page_index * element_ELEMENT_COUNT_per_page + element_index;
            ++current;
        }
    }
}

// there are 16 uint32_t per cacheline
// there are 64 (page size/cacheline size) cacheline per page
// there are ELEMENT_COUNT / 1024 pages
void separated_by_a_page_and_cacheline(uint32_t const* data,
                                       uint32_t* positions) {
    linear(data, positions);
    constexpr int elements_per_cacheline = CACHELINE_SIZE / sizeof(uint32_t);
    constexpr int elements_per_page = PAGE_SIZE / sizeof(uint32_t);
    constexpr int cacheline_per_page = PAGE_SIZE / CACHELINE_SIZE;
    constexpr int page_ELEMENT_COUNT = ELEMENT_COUNT / elements_per_page;

    static_assert(ELEMENT_COUNT % elements_per_page == 0);

    int current = 0;
    for (int element_index_in_cacheline = 0;
         element_index_in_cacheline < elements_per_cacheline;
         ++element_index_in_cacheline) {
        for (int cacheline_index_in_page = 0;
             cacheline_index_in_page < cacheline_per_page;
             ++cacheline_index_in_page) {
            for (int page_index = 0; page_index < page_ELEMENT_COUNT;
                 ++page_index) {
                positions[current++] =
                    page_index * elements_per_page +
                    cacheline_index_in_page * elements_per_cacheline +
                    element_index_in_cacheline;
            }
        }
    }
}

// there are 16 uint32_t per cacheline
// there are 64 (page size/cacheline size) cacheline per page
// there are ELEMENT_COUNT / 1024 pages
void separated_by_stride_pages_and_cacheline(uint32_t const* data,
                                             uint32_t* positions) {
    linear(data, positions);
    constexpr int elements_per_cacheline = CACHELINE_SIZE / sizeof(uint32_t);
    constexpr int elements_per_page = PAGE_SIZE / sizeof(uint32_t);
    constexpr int cacheline_per_page = PAGE_SIZE / CACHELINE_SIZE;
    constexpr int page_count = ELEMENT_COUNT / elements_per_page;

    constexpr int page_stride = STRIDE;

    static_assert(ELEMENT_COUNT % elements_per_page == 0);
    static_assert(page_stride > 0);

    int current = 0;
    for (int element_index_in_cacheline = 0;
         element_index_in_cacheline < elements_per_cacheline;
         ++element_index_in_cacheline) {
        for (int cacheline_index_in_page = 0;
             cacheline_index_in_page < cacheline_per_page;
             ++cacheline_index_in_page) {
            for (int page_start = 0;
                 page_start < page_stride && page_start < page_count;
                 ++page_start) {
                for (int page_index = page_start; page_index < page_count;
                     page_index += page_stride) {
                    positions[current++] =
                        page_index * elements_per_page +
                        cacheline_index_in_page * elements_per_cacheline +
                        element_index_in_cacheline;
                }
            }
        }
    }
}

void separated_by_stride_pages_and_cacheline_and_page_table_entry(
    uint32_t const* data, uint32_t* positions) {
    linear(data, positions);
    constexpr int elements_per_cacheline = CACHELINE_SIZE / sizeof(uint32_t);
    constexpr int elements_per_page = PAGE_SIZE / sizeof(uint32_t);
    constexpr int cacheline_per_page = PAGE_SIZE / CACHELINE_SIZE;
    constexpr int page_count = ELEMENT_COUNT / elements_per_page;

    constexpr int page_stride = STRIDE;

    constexpr int pte_size = 8;

    static_assert(ELEMENT_COUNT % elements_per_page == 0);
    static_assert(page_stride > 0);
    static_assert(pte_size <= page_stride);
    static_assert((page_stride % pte_size) == 0);

    int current = 0;
    for (int element_index_in_cacheline = 0;
         element_index_in_cacheline < elements_per_cacheline;
         ++element_index_in_cacheline) {
        for (int cacheline_index_in_page = 0;
             cacheline_index_in_page < cacheline_per_page;
             ++cacheline_index_in_page) {
            for (int pte_index = 0; pte_index < pte_size; ++pte_index) {
                for (int page_start = pte_index;
                     page_start < page_stride && page_start < page_count;
                     page_start += pte_size) {
                    for (int page_index = page_start; page_index < page_count;
                         page_index += page_stride) {
                        positions[current++] =
                            page_index * elements_per_page +
                            cacheline_index_in_page * elements_per_cacheline +
                            element_index_in_cacheline;
                    }
                }
            }
        }
    }
}

uint64_t dram_bank_key(uint64_t physical_addr) {
    uint64_t key = physical_addr & DRAM_BANK_MASK;
#if DRAM_BANK_XOR_SHIFT
    key ^= (physical_addr >> DRAM_BANK_XOR_SHIFT) & DRAM_BANK_MASK;
#endif
    return key;
}

uint64_t dram_row_key(uint64_t physical_addr) {
    return physical_addr >> DRAM_ROW_SHIFT;
}

bool physical_address(uintptr_t virtual_addr, uint64_t* physical_addr) {
    static int pagemap_fd = -2;
    if (pagemap_fd == -2) {
        pagemap_fd = open("/proc/self/pagemap", O_RDONLY);
    }

    if (pagemap_fd < 0) {
        return false;
    }

    uint64_t entry = 0;
    off_t offset = (virtual_addr / PAGE_SIZE) * sizeof(entry);
    ssize_t bytes_read = pread(pagemap_fd, &entry, sizeof(entry), offset);
    if (bytes_read != sizeof(entry)) {
        return false;
    }

    constexpr uint64_t present = 1ULL << 63;
    constexpr uint64_t pfn_mask = (1ULL << 55) - 1ULL;
    uint64_t pfn = entry & pfn_mask;
    if ((entry & present) == 0 || pfn == 0) {
        return false;
    }

    *physical_addr = (pfn * PAGE_SIZE) + (virtual_addr & (PAGE_SIZE - 1));
    return true;
}

bool collect_physical_pages(uint32_t const* data,
                            std::vector<uint64_t>* physical_pages) {
    constexpr int elements_per_page = PAGE_SIZE / sizeof(uint32_t);
    constexpr int page_count = ELEMENT_COUNT / elements_per_page;
    static_assert(ELEMENT_COUNT % elements_per_page == 0);

    physical_pages->resize(page_count);
    uintptr_t base = reinterpret_cast<uintptr_t>(data);
    for (int page_index = 0; page_index < page_count; ++page_index) {
        uint64_t physical = 0;
        uintptr_t virtual_addr =
            base + static_cast<uintptr_t>(page_index) * PAGE_SIZE;
        if (!physical_address(virtual_addr, &physical)) {
            return false;
        }
        (*physical_pages)[page_index] = physical;
    }
    return true;
}

struct RowBucket {
    uint64_t row = 0;
    std::vector<uint32_t> pages;
    size_t next = 0;
};

struct BankBucket {
    uint64_t bank = 0;
    std::vector<RowBucket> rows;
    size_t page_count = 0;
};

std::vector<BankBucket> build_bank_buckets_for_stride_start(
    std::vector<uint64_t> const& physical_pages, int page_start) {
    constexpr int elements_per_page = PAGE_SIZE / sizeof(uint32_t);
    constexpr int page_count = ELEMENT_COUNT / elements_per_page;
    constexpr int page_stride = 8;

    std::unordered_map<uint64_t,
                       std::unordered_map<uint64_t, std::vector<uint32_t>>>
        pages_by_bank_and_row;
    for (int page_index = page_start; page_index < page_count;
         page_index += page_stride) {
        uint64_t physical = physical_pages[page_index];
        pages_by_bank_and_row[dram_bank_key(physical)][dram_row_key(physical)]
            .push_back(page_index);
    }

    std::vector<BankBucket> banks;
    banks.reserve(pages_by_bank_and_row.size());
    for (auto& bank_entry : pages_by_bank_and_row) {
        BankBucket bank;
        bank.bank = bank_entry.first;
        bank.rows.reserve(bank_entry.second.size());
        for (auto& row_entry : bank_entry.second) {
            RowBucket row;
            row.row = row_entry.first;
            row.pages = std::move(row_entry.second);
            bank.page_count += row.pages.size();
            bank.rows.push_back(std::move(row));
        }

        std::sort(bank.rows.begin(), bank.rows.end(),
                  [](RowBucket const& left, RowBucket const& right) {
                      if (left.pages.size() != right.pages.size()) {
                          return left.pages.size() > right.pages.size();
                      }
                      return left.row < right.row;
                  });
        banks.push_back(std::move(bank));
    }

    std::sort(banks.begin(), banks.end(),
              [](BankBucket const& left, BankBucket const& right) {
                  if (left.page_count != right.page_count) {
                      return left.page_count > right.page_count;
                  }
                  return left.bank < right.bank;
              });

    return banks;
}

void append_same_bank_different_row_pages(std::vector<BankBucket>* banks,
                                          std::vector<uint32_t>* page_order) {
    for (BankBucket& bank : *banks) {
        size_t emitted = 0;
        while (emitted < bank.page_count) {
            for (RowBucket& row : bank.rows) {
                if (row.next >= row.pages.size()) {
                    continue;
                }
                page_order->push_back(row.pages[row.next++]);
                ++emitted;
            }
        }
    }
}

bool build_stride8_bank_conflict_page_order(
    std::vector<uint64_t> const& address_pages,
    std::vector<uint32_t>* page_order) {
    constexpr int elements_per_page = PAGE_SIZE / sizeof(uint32_t);
    constexpr int page_count = ELEMENT_COUNT / elements_per_page;
    constexpr int page_stride = 8;

    page_order->clear();
    page_order->reserve(page_count);
    size_t bank_groups = 0;
    size_t row_groups = 0;

    for (int page_start = 0; page_start < page_stride; ++page_start) {
        std::vector<BankBucket> banks =
            build_bank_buckets_for_stride_start(address_pages, page_start);
        bank_groups += banks.size();
        for (BankBucket const& bank : banks) {
            row_groups += bank.rows.size();
        }
        append_same_bank_different_row_pages(&banks, page_order);
    }

    if (page_order->size() != page_count) {
        return false;
    }

    fprintf(stderr,
            "stride8_bank_conflict: row_shift=%d bank_mask=0x%llx "
            "bank_xor_shift=%d bank_groups=%zu row_groups=%zu\n",
            DRAM_ROW_SHIFT, static_cast<unsigned long long>(DRAM_BANK_MASK),
            DRAM_BANK_XOR_SHIFT, bank_groups, row_groups);
    return true;
}

bool build_stride8_bank_conflict_physical_page_order(
    uint32_t const* data, std::vector<uint32_t>* page_order) {
    std::vector<uint64_t> physical_pages;
    if (!collect_physical_pages(data, &physical_pages)) {
        return false;
    }
    return build_stride8_bank_conflict_page_order(physical_pages, page_order);
}

void build_stride8_bank_conflict_virtual_heuristic_page_order(
    uint32_t const* data, std::vector<uint32_t>* page_order) {
    constexpr int elements_per_page = PAGE_SIZE / sizeof(uint32_t);
    constexpr int page_count = ELEMENT_COUNT / elements_per_page;

    std::vector<uint64_t> virtual_pages(page_count);
    uintptr_t base = reinterpret_cast<uintptr_t>(data);
    for (int page_index = 0; page_index < page_count; ++page_index) {
        virtual_pages[page_index] =
            base + static_cast<uintptr_t>(page_index) * PAGE_SIZE;
    }
    bool ok = build_stride8_bank_conflict_page_order(virtual_pages, page_order);
    assert(ok);
}

void separated_by_stride8_bank_conflicts_and_cacheline(uint32_t const* data,
                                                       uint32_t* positions) {
    constexpr int elements_per_cacheline = CACHELINE_SIZE / sizeof(uint32_t);
    constexpr int elements_per_page = PAGE_SIZE / sizeof(uint32_t);
    constexpr int cacheline_per_page = PAGE_SIZE / CACHELINE_SIZE;

    static_assert(ELEMENT_COUNT % elements_per_page == 0);

    std::vector<uint32_t> page_order;
    if (!build_stride8_bank_conflict_physical_page_order(data, &page_order)) {
        fprintf(stderr,
                "stride8_bank_conflict: cannot read physical PFNs from "
                "/proc/self/pagemap; using virtual-address heuristic\n");
        build_stride8_bank_conflict_virtual_heuristic_page_order(data,
                                                                 &page_order);
    }

    int current = 0;
    for (int element_index_in_cacheline = 0;
         element_index_in_cacheline < elements_per_cacheline;
         ++element_index_in_cacheline) {
        for (int cacheline_index_in_page = 0;
             cacheline_index_in_page < cacheline_per_page;
             ++cacheline_index_in_page) {
            for (uint32_t page_index : page_order) {
                positions[current++] =
                    page_index * elements_per_page +
                    cacheline_index_in_page * elements_per_cacheline +
                    element_index_in_cacheline;
            }
        }
    }
}

#define BENCHMARK_ACCESS_PATTERN(arrange_positions)               \
    do {                                                          \
        arrange_positions(data, positions);                       \
        uint64_t start = rdtsc_start();                           \
        uint32_t sum = accumulator(data, positions);              \
        uint64_t end = rdtsc_end();                               \
        print_cycles(#arrange_positions "_cycles:", end - start); \
        assert(sum == linear_scan_sum);                           \
        do_not_optimize(sum);                                     \
    } while (0)

int main(int argc, char** argv) {
    uint32_t* data = nullptr;
    uint32_t* positions = nullptr;
    int data_error =
        posix_memalign(reinterpret_cast<void**>(&data), ALLOCATION_ALIGNMENT,
                       ELEMENT_COUNT * sizeof(uint32_t));
    int positions_error =
        posix_memalign(reinterpret_cast<void**>(&positions),
                       ALLOCATION_ALIGNMENT, ELEMENT_COUNT * sizeof(uint32_t));

    if (data_error != 0 || positions_error != 0) {
        fprintf(stderr, "posix_memalign: data=%s positions=%s\n",
                strerror(data_error), strerror(positions_error));
        free(data);
        free(positions);
        return 1;
    }

    fill_data(data);
    linear(data, positions);

    // use this to validate output of all the different access patterns.
    uint32_t linear_scan_sum = accumulator(data, positions);

    bool run_all = argc == 1;
    bool run_stride8 =
        run_all || (argc == 2 && strcmp(argv[1], "stride8") == 0);
    bool run_bank = run_all || (argc == 2 && strcmp(argv[1], "bank") == 0);
    if (!run_stride8 && !run_bank) {
        fprintf(stderr, "usage: %s [stride8|bank]\n", argv[0]);
        free(data);
        free(positions);
        return 2;
    }

    if (run_stride8) {
        printf("stride=%d\n", STRIDE);
        BENCHMARK_ACCESS_PATTERN(separated_by_stride_pages_and_cacheline);
    }
    if (run_bank) {
        BENCHMARK_ACCESS_PATTERN(
            separated_by_stride8_bank_conflicts_and_cacheline);
    }
    free(data);
    free(positions);
    return 0;
}
