#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
constexpr int PAGE_SIZE = 4096;
constexpr int CACHELINE_SIZE = 64;
constexpr int ELEMENT_COUNT = 1024 * 65536; // 65M

#ifndef STRIDE
#define STRIDE 8
#endif

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

void linear(uint32_t* positions) {
    for (uint32_t i = 0; i < ELEMENT_COUNT; ++i) {
        positions[i] = i;
    }
}

void fisher_yates_shuffle(uint32_t* positions) {
    linear(positions);
    uint32_t remaining = ELEMENT_COUNT;
    for (uint32_t i = 0; i < ELEMENT_COUNT; ++i) {
        uint32_t random = rand() % remaining;
        uint32_t tmp = positions[i];
        positions[i] = positions[i + random];
        positions[i + random] = tmp;
        --remaining;
    }
}

void separated_by_a_cacheline(uint32_t* positions) {
    linear(positions);
    constexpr int element_ELEMENT_COUNT_per_cacheline = CACHELINE_SIZE / sizeof(uint32_t);
    constexpr int cacheline_ELEMENT_COUNT = ELEMENT_COUNT / element_ELEMENT_COUNT_per_cacheline;
    static_assert(ELEMENT_COUNT % element_ELEMENT_COUNT_per_cacheline == 0);
    int current = 0;
    for (int element_index = 0; element_index < element_ELEMENT_COUNT_per_cacheline; ++element_index) {
        for (int cacheline_index = 0; cacheline_index < cacheline_ELEMENT_COUNT; ++cacheline_index) {
            positions[current] = cacheline_index * element_ELEMENT_COUNT_per_cacheline + element_index;
            ++current;
        }
    }
}

void separated_by_a_page(uint32_t* positions) {
    linear(positions);
    constexpr int element_ELEMENT_COUNT_per_page = PAGE_SIZE / sizeof(uint32_t);
    constexpr int page_ELEMENT_COUNT = ELEMENT_COUNT / element_ELEMENT_COUNT_per_page;
    static_assert(ELEMENT_COUNT % element_ELEMENT_COUNT_per_page == 0);
    int current = 0;
    for (int element_index = 0; element_index < element_ELEMENT_COUNT_per_page; ++element_index) {
        for (int page_index = 0; page_index < page_ELEMENT_COUNT; ++page_index) {
            positions[current] = page_index * element_ELEMENT_COUNT_per_page + element_index;
            ++current;
        }
    }
}

// there are 16 uint32_t per cacheline
// there are 64 (page size/cacheline size) cacheline per page
// there are ELEMENT_COUNT / 1024 pages
void separated_by_a_page_and_cacheline(uint32_t* positions) {
    linear(positions);
    constexpr int elements_per_cacheline = CACHELINE_SIZE / sizeof(uint32_t);
    constexpr int elements_per_page = PAGE_SIZE / sizeof(uint32_t);
    constexpr int cacheline_per_page = PAGE_SIZE / CACHELINE_SIZE;
    constexpr int page_ELEMENT_COUNT = ELEMENT_COUNT / elements_per_page;

    static_assert(ELEMENT_COUNT % elements_per_page == 0);

    int current = 0;
    for (int element_index_in_cacheline = 0; element_index_in_cacheline < elements_per_cacheline; ++element_index_in_cacheline) {
        for (int cacheline_index_in_page = 0; cacheline_index_in_page < cacheline_per_page; ++cacheline_index_in_page) {
            for (int page_index = 0; page_index < page_ELEMENT_COUNT; ++page_index) {
                positions[current++] = page_index * elements_per_page + cacheline_index_in_page * elements_per_cacheline + element_index_in_cacheline;
            }
        }
    }
}

// there are 16 uint32_t per cacheline
// there are 64 (page size/cacheline size) cacheline per page
// there are ELEMENT_COUNT / 1024 pages
void separated_by_stride_pages_and_cacheline(uint32_t* positions) {
    linear(positions);
    constexpr int elements_per_cacheline = CACHELINE_SIZE / sizeof(uint32_t);
    constexpr int elements_per_page = PAGE_SIZE / sizeof(uint32_t);
    constexpr int cacheline_per_page = PAGE_SIZE / CACHELINE_SIZE;
    constexpr int page_ELEMENT_COUNT = ELEMENT_COUNT / elements_per_page;

    constexpr int page_stride = STRIDE;

    static_assert(ELEMENT_COUNT % elements_per_page == 0);
    static_assert(page_stride > 0);

    int current = 0;
    for (int element_index_in_cacheline = 0; element_index_in_cacheline < elements_per_cacheline; ++element_index_in_cacheline) {
        for (int cacheline_index_in_page = 0; cacheline_index_in_page < cacheline_per_page; ++cacheline_index_in_page) {
            for (int page_start = 0; page_start < page_stride && page_start < page_ELEMENT_COUNT; ++page_start) {
                for (int page_index = page_start; page_index < page_ELEMENT_COUNT; page_index += page_stride) {
                    positions[current++] = page_index * elements_per_page + cacheline_index_in_page * elements_per_cacheline + element_index_in_cacheline;
                }
            }
        }
    }
}


#define BENCHMARK_ACCESS_PATTERN(arrange_positions)                     \
    do {                                                                \
        arrange_positions(positions);                                   \
        uint64_t start = rdtsc_start();                                 \
        uint32_t sum = accumulator(data, positions);                     \
        uint64_t end = rdtsc_end();                                     \
        print_cycles(#arrange_positions "_cycles:", end - start);       \
        assert(sum == linear_scan_sum);                                 \
        do_not_optimize(sum);                                           \
    } while (0)

int main() {
    uint32_t *data = (uint32_t*)calloc(ELEMENT_COUNT, sizeof(uint32_t));
    uint32_t *positions = (uint32_t*)calloc(ELEMENT_COUNT, sizeof(uint32_t));

    if (!data || !positions) {
        perror("calloc");
        return 1;
    }

    fill_data(data);
    linear(positions);

    // use this to validate output of all the different access patterns.
    uint32_t linear_scan_sum = accumulator(data, positions);

    BENCHMARK_ACCESS_PATTERN(linear);
    BENCHMARK_ACCESS_PATTERN(fisher_yates_shuffle);
    BENCHMARK_ACCESS_PATTERN(separated_by_a_cacheline);
    BENCHMARK_ACCESS_PATTERN(separated_by_a_page);
    BENCHMARK_ACCESS_PATTERN(separated_by_a_page_and_cacheline);
    printf("stride=%d\n", STRIDE);
    BENCHMARK_ACCESS_PATTERN(separated_by_stride_pages_and_cacheline);
    return 0;
}
