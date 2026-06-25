#include <fcntl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

#ifndef STRIDE
#define STRIDE 8
#endif

// Machine constants
constexpr int PAGE_SIZE = 4096;
constexpr int CACHELINE_SIZE = 64;

// element usage count that we are using
constexpr int ELEMENT_COUNT = (1 << 16) * (PAGE_SIZE / sizeof(uint32_t));

// Assumed DRAM config
constexpr size_t skip = 4096;  // emprically tested between 512 to 8192
constexpr uint32_t DRAM_BANK_GROUP_COUNT = 4;
constexpr uint32_t DRAM_BANK_COUNT_PER_GROUP = 4;
constexpr uint32_t DRAM_ROW_SHIFT = 16;  // emprically tested between 15 to 19
constexpr int PAGE_STRIDE = 8;           // pin to 8 because graph say 8 is best

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
    printf("%-58s %12lu\n", label, cycles);
}

void do_not_optimize(uint32_t value) {
    __asm__ __volatile__("" : : "r,m"(value) : "memory");
}

bool allocate_without_huge_pages(char const* name, uint32_t** buffer) {
    constexpr size_t ALLOCATION_BYTES =
        static_cast<size_t>(ELEMENT_COUNT) * sizeof(uint32_t);
    void* memory = nullptr;
    int error = posix_memalign(&memory, PAGE_SIZE, ALLOCATION_BYTES);
    if (error != 0) {
        fprintf(stderr, "posix_memalign %s: %s\n", name, strerror(error));
        return false;
    }

    if (madvise(memory, ALLOCATION_BYTES, MADV_NOHUGEPAGE) != 0) {
        fprintf(stderr, "madvise MADV_NOHUGEPAGE %s: %s\n", name,
                strerror(errno));
        free(memory);
        return false;
    }

    *buffer = static_cast<uint32_t*>(memory);
    return true;
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

void reset(uint32_t const*, uint32_t* positions) {
    for (uint32_t i = 0; i < ELEMENT_COUNT; ++i) {
        positions[i] = -1;
    }
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
    constexpr int element_count_per_cacheline =
        CACHELINE_SIZE / sizeof(uint32_t);
    constexpr int cacheline_count = ELEMENT_COUNT / element_count_per_cacheline;
    static_assert(ELEMENT_COUNT % element_count_per_cacheline == 0);
    int current = 0;
    for (int element_index = 0; element_index < element_count_per_cacheline;
         ++element_index) {
        for (int cacheline_index = 0; cacheline_index < cacheline_count;
             ++cacheline_index) {
            positions[current] =
                cacheline_index * element_count_per_cacheline + element_index;
            ++current;
        }
    }
}

void separated_by_a_page(uint32_t const* data, uint32_t* positions) {
    constexpr int element_count_per_page = PAGE_SIZE / sizeof(uint32_t);
    constexpr int page_count = ELEMENT_COUNT / element_count_per_page;
    static_assert(ELEMENT_COUNT % element_count_per_page == 0);
    int current = 0;
    for (int element_index = 0; element_index < element_count_per_page;
         ++element_index) {
        for (int page_index = 0; page_index < page_count; ++page_index) {
            positions[current] =
                page_index * element_count_per_page + element_index;
            ++current;
        }
    }
}

void separated_by_a_page_and_cacheline(uint32_t const* data,
                                       uint32_t* positions) {
    constexpr int elements_per_cacheline = CACHELINE_SIZE / sizeof(uint32_t);
    constexpr int elements_per_page = PAGE_SIZE / sizeof(uint32_t);
    constexpr int cacheline_per_page = PAGE_SIZE / CACHELINE_SIZE;
    constexpr int page_count = ELEMENT_COUNT / elements_per_page;

    static_assert(ELEMENT_COUNT % elements_per_page == 0);

    int current = 0;
    for (int element_index_in_cacheline = 0;
         element_index_in_cacheline < elements_per_cacheline;
         ++element_index_in_cacheline) {
        for (int cacheline_index_in_page = 0;
             cacheline_index_in_page < cacheline_per_page;
             ++cacheline_index_in_page) {
            for (int page_index = 0; page_index < page_count; ++page_index) {
                positions[current++] =
                    page_index * elements_per_page +
                    cacheline_index_in_page * elements_per_cacheline +
                    element_index_in_cacheline;
            }
        }
    }
}

template <int page_stride>
void separated_by_stride_pages_and_cacheline(uint32_t const* data,
                                             uint32_t* positions) {
    constexpr int elements_per_cacheline = CACHELINE_SIZE / sizeof(uint32_t);
    constexpr int elements_per_page = PAGE_SIZE / sizeof(uint32_t);
    constexpr int cacheline_per_page = PAGE_SIZE / CACHELINE_SIZE;
    constexpr int page_count = ELEMENT_COUNT / elements_per_page;

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

// void separated_by_stride_pages_and_cacheline_and_page_table_entry(
//     uint32_t const* data, uint32_t* positions) {
//     constexpr int elements_per_cacheline = CACHELINE_SIZE / sizeof(uint32_t);
//     constexpr int elements_per_page = PAGE_SIZE / sizeof(uint32_t);
//     constexpr int cacheline_per_page = PAGE_SIZE / CACHELINE_SIZE;
//     constexpr int page_count = ELEMENT_COUNT / elements_per_page;

//     constexpr int page_stride = STRIDE;

//     constexpr int pte_size = 8;

//     static_assert(ELEMENT_COUNT % elements_per_page == 0);
//     static_assert(page_stride > 0);
//     static_assert(pte_size <= page_stride);
//     static_assert((page_stride % pte_size) == 0);

//     int current = 0;
//     for (int element_index_in_cacheline = 0;
//          element_index_in_cacheline < elements_per_cacheline;
//          ++element_index_in_cacheline) {
//         for (int cacheline_index_in_page = 0;
//              cacheline_index_in_page < cacheline_per_page;
//              ++cacheline_index_in_page) {
//             for (int pte_index = 0; pte_index < pte_size; ++pte_index) {
//                 for (int page_start = pte_index;
//                      page_start < page_stride && page_start < page_count;
//                      page_start += pte_size) {
//                     for (int page_index = page_start; page_index <
//                     page_count;
//                          page_index += page_stride) {
//                         positions[current++] =
//                             page_index * elements_per_page +
//                             cacheline_index_in_page * elements_per_cacheline
//                             + element_index_in_cacheline;
//                     }
//                 }
//             }
//         }
//     }
// }

namespace ram_code {
// virtual address to physical address, courtesy of Codex.
uint64_t physical_address(uintptr_t virtual_addr) {
    static int pagemap_fd = -2;
    if (pagemap_fd == -2) {
        pagemap_fd = open("/proc/self/pagemap", O_RDONLY);
    }

    if (pagemap_fd < 0) {
        throw std::runtime_error("cant open pagemap");
    }

    uint64_t entry = 0;
    off_t offset = (virtual_addr / PAGE_SIZE) * sizeof(entry);
    ssize_t bytes_read = pread(pagemap_fd, &entry, sizeof(entry), offset);
    if (bytes_read != sizeof(entry)) {
        throw std::runtime_error("bad PTE read");
    }

    constexpr uint64_t present = 1ULL << 63;
    constexpr uint64_t pfn_mask = (1ULL << 55) - 1ULL;
    uint64_t pfn = entry & pfn_mask;
    if ((entry & present) == 0 || pfn == 0) {
        throw std::runtime_error("present page is marked not present");
    }

    return (pfn * PAGE_SIZE) + (virtual_addr & (PAGE_SIZE - 1));
}

struct DramLocation {
    uint64_t bank_index;
    uint64_t rank;     // always assume 0
    uint64_t channel;  // always assume 0
    uint64_t row_index;
    uint32_t page_index;
};

DramLocation physical_address_to_dram_location(uint64_t physical_address,
                                               uint32_t page_index) {
    auto get_bit = [&](uint32_t index) {
        return (physical_address >> index) & 1;
    };

    uint64_t bg0 = get_bit(7) ^ get_bit(14);
    uint64_t bg1 = get_bit(15) ^ get_bit(19);
    uint64_t bg = bg1 * 2 + bg0;
    uint64_t ba0 = get_bit(17) ^ get_bit(21);
    uint64_t ba1 = get_bit(18) ^ get_bit(22);
    uint64_t ba = ba1 * 2 + ba0;

    return {
        .bank_index = bg * 4 + ba,
        .rank = 0,
        .channel = 0,
        .row_index = physical_address >> DRAM_ROW_SHIFT,
        .page_index = page_index,
    };
}

// pages to stride buckets + stride buckets to physical address/DramLocation
// within each stride_bucket, group them into distinct bank_index
// within each bank_index, round-robin the rows

// map each page in data to DramLocation, and bucketize them into the stride
// that they are in (all (page_index % 8) are in same bucket).
std::array<std::vector<DramLocation>, PAGE_STRIDE> bucketize_pages_from_stride(
    uint32_t const* data) {
    std::array<std::vector<DramLocation>, PAGE_STRIDE> result;
    constexpr int elements_per_page = PAGE_SIZE / sizeof(uint32_t);
    constexpr int page_count = ELEMENT_COUNT / elements_per_page;

    static_assert(ELEMENT_COUNT % elements_per_page == 0);
    static_assert((page_count % PAGE_STRIDE) == 0);

    for (int page_stride_index = 0; page_stride_index < PAGE_STRIDE;
         ++page_stride_index) {
        for (int i = 0; i < (page_count / PAGE_STRIDE); ++i) {
            uint32_t page_index = i * PAGE_STRIDE + page_stride_index;
            DramLocation dram_location = physical_address_to_dram_location(
                physical_address(reinterpret_cast<uintptr_t>(
                    data + page_index * elements_per_page)),
                page_index);
            result[page_stride_index].push_back(dram_location);
        }
    }
    return result;
}

constexpr uint32_t DRAM_BANK_COUNT =
    DRAM_BANK_GROUP_COUNT * DRAM_BANK_COUNT_PER_GROUP;

// for each stride bucket (elements in the same stride), bucketize them into the
// bank index that they are in, then sort each bucket such that elements are
// arranged by the rows that they are in.
std::array<std::vector<DramLocation>, DRAM_BANK_COUNT> bucket_into_bank_index(
    std::vector<DramLocation>& stride_bucket) {
    std::array<std::vector<DramLocation>, DRAM_BANK_COUNT> result;
    for (DramLocation const& dram_location : stride_bucket) {
        result[dram_location.bank_index].push_back(dram_location);
    }
    for (int i = 0; i < 16; ++i) {
        std::ranges::sort(result[i], {}, &DramLocation::row_index);
    }
    return result;
}

// my shit round-robin, lazily stride?
void round_robin_row_bucket(std::vector<DramLocation>& data_from_same_bank) {
    std::vector<DramLocation> result;
    result.reserve(data_from_same_bank.size());
    for (std::size_t stride_index = 0;
         stride_index < skip && stride_index < data_from_same_bank.size();
         ++stride_index) {
        for (std::size_t i = stride_index; i < data_from_same_bank.size();
             i += skip) {
            result.push_back(data_from_same_bank[i]);
        }
    }
    data_from_same_bank = std::move(result);
}

std::vector<uint32_t> build_stride8_bank_conflict_physical_page_order(
    uint32_t const* data) {
    // TODO don't hardcode 16
    std::vector<uint32_t> result;
    constexpr int elements_per_page = PAGE_SIZE / sizeof(uint32_t);
    constexpr int page_count = ELEMENT_COUNT / elements_per_page;
    result.reserve(page_count);
    std::array<std::vector<DramLocation>, PAGE_STRIDE> buckets =
        bucketize_pages_from_stride(data);
    for (int stride = 0; stride < PAGE_STRIDE; ++stride) {
        std::array<std::vector<DramLocation>, DRAM_BANK_COUNT>
            bank_index_buckets = bucket_into_bank_index(buckets[stride]);
        for (int bank_index = 0; bank_index < DRAM_BANK_COUNT; ++bank_index) {
            round_robin_row_bucket(bank_index_buckets[bank_index]);
            for (DramLocation& l : bank_index_buckets[bank_index]) {
                result.push_back(l.page_index);
            }
        }
    }
    return result;
}
}  // namespace ram_code

void separated_by_stride8_bank_conflicts_and_cacheline(uint32_t const* data,
                                                       uint32_t* positions) {
    constexpr int elements_per_cacheline = CACHELINE_SIZE / sizeof(uint32_t);
    constexpr int elements_per_page = PAGE_SIZE / sizeof(uint32_t);
    constexpr int cacheline_per_page = PAGE_SIZE / CACHELINE_SIZE;

    static_assert(ELEMENT_COUNT % elements_per_page == 0);

    std::vector<uint32_t> page_order =
        ram_code::build_stride8_bank_conflict_physical_page_order(data);
    assert(page_order.size() == ELEMENT_COUNT / elements_per_page);

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
    assert(current == ELEMENT_COUNT);
}

// call linear at start to reset positions - some arrangements (eg shuffle)
// might rely on this.
#define BENCHMARK_ACCESS_PATTERN(arrange_positions)               \
    do {                                                          \
        reset(data, positions);                                   \
        arrange_positions(data, positions);                       \
        uint64_t start = rdtsc_start();                           \
        uint32_t sum = accumulator(data, positions);              \
        uint64_t end = rdtsc_end();                               \
        print_cycles(#arrange_positions "_cycles:", end - start); \
        assert(sum == linear_scan_sum);                           \
        do_not_optimize(sum);                                     \
    } while (0)

int main() {
    uint32_t* data = nullptr;
    uint32_t* positions = nullptr;

    if (!allocate_without_huge_pages("data", &data) ||
        !allocate_without_huge_pages("positions", &positions)) {
        free(data);
        free(positions);
        return 1;
    }

    fill_data(data);
    linear(data, positions);

    // use this to validate output of all the different access patterns.
    uint32_t linear_scan_sum = accumulator(data, positions);

    BENCHMARK_ACCESS_PATTERN(linear);
    BENCHMARK_ACCESS_PATTERN(fisher_yates_shuffle);
    BENCHMARK_ACCESS_PATTERN(separated_by_a_cacheline);
    BENCHMARK_ACCESS_PATTERN(separated_by_a_page);
    BENCHMARK_ACCESS_PATTERN(separated_by_a_page_and_cacheline);
    printf("stride used=%d ", STRIDE);
    BENCHMARK_ACCESS_PATTERN(separated_by_stride_pages_and_cacheline<STRIDE>);
    BENCHMARK_ACCESS_PATTERN(separated_by_stride8_bank_conflicts_and_cacheline);

    free(data);
    free(positions);
    return 0;
}
