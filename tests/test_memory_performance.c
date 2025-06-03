/*
 * ═══════════════════════════════════════════════════════════════════════════════════
 *                         ETHEREUM MEV RESEARCH - MEMORY TEST SUITE
 *                              Memory Management Performance Testing
 *                                        C Implementation
 * ═══════════════════════════════════════════════════════════════════════════════════
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <assert.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/time.h>

// Include our memory utilities
#include "../src/utils/memory_utils.c"

#define TEST_ITERATIONS 100000
#define THREAD_COUNT 8
#define ALLOCATIONS_PER_THREAD 10000
#define STRESS_TEST_DURATION 30 // seconds

// Performance measurement structure
typedef struct {
    uint64_t total_time_ns;
    uint64_t min_time_ns;
    uint64_t max_time_ns;
    uint64_t *measurements;
    size_t count;
} perf_stats_t;

// Get high-precision timestamp
static uint64_t get_timestamp_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

// Calculate percentile from sorted array
static uint64_t calculate_percentile(uint64_t *sorted_array, size_t count, double percentile) {
    size_t index = (size_t)(count * percentile / 100.0);
    if (index >= count) index = count - 1;
    return sorted_array[index];
}

// Comparison function for qsort
static int compare_uint64(const void *a, const void *b) {
    uint64_t ua = *(const uint64_t*)a;
    uint64_t ub = *(const uint64_t*)b;
    return (ua > ub) - (ua < ub);
}

// Print performance statistics
static void print_perf_stats(const char *test_name, perf_stats_t *stats) {
    if (stats->count == 0) return;
    
    // Sort measurements for percentile calculations
    qsort(stats->measurements, stats->count, sizeof(uint64_t), compare_uint64);
    
    uint64_t mean = stats->total_time_ns / stats->count;
    uint64_t median = calculate_percentile(stats->measurements, stats->count, 50.0);
    uint64_t p95 = calculate_percentile(stats->measurements, stats->count, 95.0);
    uint64_t p99 = calculate_percentile(stats->measurements, stats->count, 99.0);
    uint64_t p999 = calculate_percentile(stats->measurements, stats->count, 99.9);
    
    double throughput = stats->count / (stats->total_time_ns / 1e9);
    
    printf("\n🧠 Memory Performance: %s\n", test_name);
    printf("   Iterations:     %zu\n", stats->count);
    printf("   Mean latency:   %lu ns (%.3f μs)\n", mean, mean / 1000.0);
    printf("   Median latency: %lu ns (%.3f μs)\n", median, median / 1000.0);
    printf("   Min latency:    %lu ns\n", stats->min_time_ns);
    printf("   Max latency:    %lu ns\n", stats->max_time_ns);
    printf("   P95 latency:    %lu ns (%.3f μs)\n", p95, p95 / 1000.0);
    printf("   P99 latency:    %lu ns (%.3f μs)\n", p99, p99 / 1000.0);
    printf("   P99.9 latency:  %lu ns (%.3f μs)\n", p999, p999 / 1000.0);
    printf("   Throughput:     %.0f ops/sec\n", throughput);
    
    // Performance assertions
    assert(mean < 50000); // 50 μs mean allocation time
    assert(p99 < 200000);  // 200 μs P99 allocation time
    assert(throughput > 100000); // 100k ops/sec minimum
}

// Test basic allocation/deallocation performance
void test_basic_allocation_performance() {
    printf("\n🚀 Testing Basic Allocation Performance\n");
    
    perf_stats_t stats = {0};
    stats.measurements = malloc(TEST_ITERATIONS * sizeof(uint64_t));
    stats.count = TEST_ITERATIONS;
    stats.min_time_ns = UINT64_MAX;
    stats.max_time_ns = 0;
    
    void **ptrs = malloc(TEST_ITERATIONS * sizeof(void*));
    
    // Test allocation performance
    for (size_t i = 0; i < TEST_ITERATIONS; i++) {
        size_t size = 64 + (i % 1024); // Variable sizes
        
        uint64_t start = get_timestamp_ns();
        ptrs[i] = mev_malloc(size);
        uint64_t end = get_timestamp_ns();
        
        uint64_t duration = end - start;
        stats.measurements[i] = duration;
        stats.total_time_ns += duration;
        
        if (duration < stats.min_time_ns) stats.min_time_ns = duration;
        if (duration > stats.max_time_ns) stats.max_time_ns = duration;
        
        assert(ptrs[i] != NULL);
    }
    
    print_perf_stats("Basic Allocation", &stats);
    
    // Test deallocation performance
    memset(&stats, 0, sizeof(stats));
    stats.measurements = realloc(stats.measurements, TEST_ITERATIONS * sizeof(uint64_t));
    stats.count = TEST_ITERATIONS;
    stats.min_time_ns = UINT64_MAX;
    
    for (size_t i = 0; i < TEST_ITERATIONS; i++) {
        uint64_t start = get_timestamp_ns();
        mev_free(ptrs[i]);
        uint64_t end = get_timestamp_ns();
        
        uint64_t duration = end - start;
        stats.measurements[i] = duration;
        stats.total_time_ns += duration;
        
        if (duration < stats.min_time_ns) stats.min_time_ns = duration;
        if (duration > stats.max_time_ns) stats.max_time_ns = duration;
    }
    
    print_perf_stats("Basic Deallocation", &stats);
    
    free(stats.measurements);
    free(ptrs);
}

// Test vectorized memory operations performance
void test_vectorized_operations() {
    printf("\n⚡ Testing Vectorized Memory Operations\n");
    
    const size_t test_sizes[] = {1024, 4096, 16384, 65536, 262144, 1048576};
    const size_t num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    for (size_t i = 0; i < num_sizes; i++) {
        size_t size = test_sizes[i];
        void *src = mev_malloc(size);
        void *dst = mev_malloc(size);
        void *set_buf = mev_malloc(size);
        
        assert(src && dst && set_buf);
        
        // Fill source with test data
        memset(src, 0xAA, size);
        
        // Test vectorized memcpy
        const int copy_iterations = 1000;
        uint64_t copy_start = get_timestamp_ns();
        for (int j = 0; j < copy_iterations; j++) {
            mev_memcpy(dst, src, size);
        }
        uint64_t copy_end = get_timestamp_ns();
        
        // Test vectorized memset
        const int set_iterations = 1000;
        uint64_t set_start = get_timestamp_ns();
        for (int j = 0; j < set_iterations; j++) {
            mev_memset(set_buf, 0x55, size);
        }
        uint64_t set_end = get_timestamp_ns();
        
        double copy_throughput = (copy_iterations * size) / ((copy_end - copy_start) / 1e9) / 1e9; // GB/s
        double set_throughput = (set_iterations * size) / ((set_end - set_start) / 1e9) / 1e9; // GB/s
        
        printf("   Size %zu bytes:\n", size);
        printf("     Vectorized memcpy: %.2f GB/s\n", copy_throughput);
        printf("     Vectorized memset: %.2f GB/s\n", set_throughput);
        
        // Verify correctness
        assert(memcmp(src, dst, size) == 0);
        for (size_t k = 0; k < size; k++) {
            assert(((uint8_t*)set_buf)[k] == 0x55);
        }
        
        // Performance assertions
        assert(copy_throughput > 1.0); // At least 1 GB/s
        assert(set_throughput > 5.0);  // At least 5 GB/s
        
        mev_free(src);
        mev_free(dst);
        mev_free(set_buf);
    }
}

// Thread data for concurrent testing
typedef struct {
    int thread_id;
    size_t allocations;
    uint64_t total_time;
    uint64_t *durations;
} thread_data_t;

// Thread function for concurrent allocation testing
void *concurrent_allocation_thread(void *arg) {
    thread_data_t *data = (thread_data_t*)arg;
    void **ptrs = malloc(data->allocations * sizeof(void*));
    
    uint64_t start = get_timestamp_ns();
    
    // Allocate
    for (size_t i = 0; i < data->allocations; i++) {
        size_t size = 64 + ((data->thread_id * 1000 + i) % 1024);
        
        uint64_t alloc_start = get_timestamp_ns();
        ptrs[i] = mev_malloc(size);
        uint64_t alloc_end = get_timestamp_ns();
        
        data->durations[i] = alloc_end - alloc_start;
        assert(ptrs[i] != NULL);
        
        // Write to memory to ensure it's accessible
        memset(ptrs[i], data->thread_id, size);
    }
    
    // Deallocate
    for (size_t i = 0; i < data->allocations; i++) {
        mev_free(ptrs[i]);
    }
    
    uint64_t end = get_timestamp_ns();
    data->total_time = end - start;
    
    free(ptrs);
    return NULL;
}

// Test concurrent allocation performance
void test_concurrent_allocation() {
    printf("\n🔄 Testing Concurrent Allocation Performance\n");
    
    pthread_t threads[THREAD_COUNT];
    thread_data_t thread_data[THREAD_COUNT];
    
    // Initialize thread data
    for (int i = 0; i < THREAD_COUNT; i++) {
        thread_data[i].thread_id = i;
        thread_data[i].allocations = ALLOCATIONS_PER_THREAD;
        thread_data[i].total_time = 0;
        thread_data[i].durations = malloc(ALLOCATIONS_PER_THREAD * sizeof(uint64_t));
    }
    
    uint64_t start = get_timestamp_ns();
    
    // Create threads
    for (int i = 0; i < THREAD_COUNT; i++) {
        int rc = pthread_create(&threads[i], NULL, concurrent_allocation_thread, &thread_data[i]);
        assert(rc == 0);
    }
    
    // Wait for threads to complete
    for (int i = 0; i < THREAD_COUNT; i++) {
        pthread_join(threads[i], NULL);
    }
    
    uint64_t end = get_timestamp_ns();
    uint64_t total_duration = end - start;
    
    // Aggregate statistics
    size_t total_allocations = THREAD_COUNT * ALLOCATIONS_PER_THREAD;
    double throughput = total_allocations / (total_duration / 1e9);
    
    printf("   Threads:            %d\n", THREAD_COUNT);
    printf("   Allocations/thread: %d\n", ALLOCATIONS_PER_THREAD);
    printf("   Total allocations:  %zu\n", total_allocations);
    printf("   Total duration:     %lu ns (%.3f ms)\n", total_duration, total_duration / 1e6);
    printf("   Throughput:         %.0f ops/sec\n", throughput);
    
    // Calculate per-thread statistics
    uint64_t total_individual_time = 0;
    for (int i = 0; i < THREAD_COUNT; i++) {
        total_individual_time += thread_data[i].total_time;
        printf("   Thread %d time:     %lu ns (%.3f ms)\n", i, 
               thread_data[i].total_time, thread_data[i].total_time / 1e6);
    }
    
    double parallelism_efficiency = (double)total_individual_time / (total_duration * THREAD_COUNT);
    printf("   Parallelism efficiency: %.2f\n", parallelism_efficiency);
    
    // Performance assertions
    assert(throughput > 500000); // 500k allocations/sec minimum
    assert(parallelism_efficiency > 0.7); // 70% efficiency minimum
    
    // Clean up
    for (int i = 0; i < THREAD_COUNT; i++) {
        free(thread_data[i].durations);
    }
}

// Test memory fragmentation and pool efficiency
void test_fragmentation_resistance() {
    printf("\n📦 Testing Fragmentation Resistance\n");
    
    const size_t pattern_size = 1000;
    void **ptrs = malloc(pattern_size * sizeof(void*));
    
    // Allocation pattern that causes fragmentation in naive allocators
    printf("   Phase 1: Mixed size allocations\n");
    for (size_t i = 0; i < pattern_size; i++) {
        size_t size = (i % 3 == 0) ? 64 : ((i % 3 == 1) ? 256 : 1024);
        ptrs[i] = mev_malloc(size);
        assert(ptrs[i] != NULL);
    }
    
    memory_stats_t initial_stats = mev_get_memory_stats();
    printf("   Initial fragmentation ratio: %.4f\n", initial_stats.fragmentation_ratio);
    
    // Free every other allocation (creates holes)
    printf("   Phase 2: Create fragmentation holes\n");
    for (size_t i = 0; i < pattern_size; i += 2) {
        mev_free(ptrs[i]);
        ptrs[i] = NULL;
    }
    
    memory_stats_t fragmented_stats = mev_get_memory_stats();
    printf("   Fragmented ratio: %.4f\n", fragmented_stats.fragmentation_ratio);
    
    // Reallocate in the holes
    printf("   Phase 3: Reallocate in holes\n");
    for (size_t i = 0; i < pattern_size; i += 2) {
        ptrs[i] = mev_malloc(128); // Different size to test hole reuse
        assert(ptrs[i] != NULL);
    }
    
    memory_stats_t final_stats = mev_get_memory_stats();
    printf("   Final fragmentation ratio: %.4f\n", final_stats.fragmentation_ratio);
    
    // Clean up
    for (size_t i = 0; i < pattern_size; i++) {
        if (ptrs[i]) {
            mev_free(ptrs[i]);
        }
    }
    
    free(ptrs);
    
    // Fragmentation should be well-controlled
    assert(final_stats.fragmentation_ratio < 2.0);
    printf("   ✅ Fragmentation well-controlled\n");
}

// Test memory prefetching performance
void test_prefetch_performance() {
    printf("\n🎯 Testing Memory Prefetch Performance\n");
    
    const size_t array_size = 1024 * 1024; // 1M elements
    const size_t element_size = sizeof(uint64_t);
    const size_t stride = 64; // Cache line size
    
    uint64_t *array = (uint64_t*)mev_malloc(array_size * element_size);
    assert(array != NULL);
    
    // Initialize array
    for (size_t i = 0; i < array_size; i++) {
        array[i] = i;
    }
    
    // Test without prefetching
    uint64_t start_no_prefetch = get_timestamp_ns();
    uint64_t sum_no_prefetch = 0;
    for (size_t i = 0; i < array_size; i += stride) {
        sum_no_prefetch += array[i];
    }
    uint64_t end_no_prefetch = get_timestamp_ns();
    
    // Test with prefetching
    uint64_t start_prefetch = get_timestamp_ns();
    uint64_t sum_prefetch = 0;
    for (size_t i = 0; i < array_size; i += stride) {
        // Prefetch next cache line
        if (i + stride * 2 < array_size) {
            mev_prefetch(&array[i + stride * 2], 1);
        }
        sum_prefetch += array[i];
    }
    uint64_t end_prefetch = get_timestamp_ns();
    
    uint64_t duration_no_prefetch = end_no_prefetch - start_no_prefetch;
    uint64_t duration_prefetch = end_prefetch - start_prefetch;
    double speedup = (double)duration_no_prefetch / duration_prefetch;
    
    printf("   Array size:         %zu elements\n", array_size);
    printf("   No prefetch time:   %lu ns (%.3f ms)\n", duration_no_prefetch, duration_no_prefetch / 1e6);
    printf("   With prefetch time: %lu ns (%.3f ms)\n", duration_prefetch, duration_prefetch / 1e6);
    printf("   Speedup:            %.2fx\n", speedup);
    
    // Verify correctness
    assert(sum_no_prefetch == sum_prefetch);
    
    // Prefetching should provide some benefit
    assert(speedup > 1.0);
    
    mev_free(array);
}

// Long-running stress test
void test_memory_stress() {
    printf("\n💥 Running Memory Stress Test (%d seconds)\n", STRESS_TEST_DURATION);
    
    time_t start_time = time(NULL);
    size_t operations = 0;
    size_t active_allocations = 0;
    const size_t max_active = 10000;
    
    void **active_ptrs = malloc(max_active * sizeof(void*));
    size_t *active_sizes = malloc(max_active * sizeof(size_t));
    
    while (time(NULL) - start_time < STRESS_TEST_DURATION) {
        if (active_allocations < max_active && (rand() % 100) < 60) {
            // Allocate (60% probability)
            size_t size = 64 + (rand() % 4032); // Random size up to 4KB
            void *ptr = mev_malloc(size);
            
            if (ptr) {
                active_ptrs[active_allocations] = ptr;
                active_sizes[active_allocations] = size;
                active_allocations++;
                
                // Write to the memory
                memset(ptr, rand() % 256, size);
            }
        } else if (active_allocations > 0) {
            // Deallocate (40% probability or when at max capacity)
            size_t index = rand() % active_allocations;
            mev_free(active_ptrs[index]);
            
            // Move last element to fill the gap
            active_ptrs[index] = active_ptrs[active_allocations - 1];
            active_sizes[index] = active_sizes[active_allocations - 1];
            active_allocations--;
        }
        
        operations++;
        
        // Periodic stats check
        if (operations % 10000 == 0) {
            memory_stats_t stats = mev_get_memory_stats();
            printf("   Operations: %zu, Active: %zu, Used: %.2f MB, Fragmentation: %.4f\n",
                   operations, active_allocations, 
                   stats.used_size / 1024.0 / 1024.0,
                   stats.fragmentation_ratio);
        }
    }
    
    // Clean up remaining allocations
    for (size_t i = 0; i < active_allocations; i++) {
        mev_free(active_ptrs[i]);
    }
    
    memory_stats_t final_stats = mev_get_memory_stats();
    double ops_per_sec = operations / (double)STRESS_TEST_DURATION;
    
    printf("   Total operations:   %zu\n", operations);
    printf("   Operations/sec:     %.0f\n", ops_per_sec);
    printf("   Peak allocations:   %zu\n", max_active);
    printf("   Final stats:\n");
    printf("     Total allocated:  %lu bytes\n", final_stats.total_allocated);
    printf("     Allocation count: %lu\n", final_stats.allocation_count);
    printf("     Deallocation count: %lu\n", final_stats.deallocation_count);
    printf("     Fragmentation:    %.4f\n", final_stats.fragmentation_ratio);
    
    free(active_ptrs);
    free(active_sizes);
    
    // Stress test assertions
    assert(ops_per_sec > 10000); // 10k ops/sec minimum under stress
    assert(final_stats.fragmentation_ratio < 5.0); // Reasonable fragmentation
    assert(final_stats.allocation_count > final_stats.deallocation_count); // Some allocs remain
    
    printf("   ✅ Stress test completed successfully\n");
}

int main(void) {
    printf("════════════════════════════════════════════════════════════════════════════════════\n");
    printf("                        ETHEREUM MEV RESEARCH - MEMORY TEST SUITE\n");
    printf("                              High-Performance Memory Testing\n");
    printf("════════════════════════════════════════════════════════════════════════════════════\n");
    
    // Initialize random seed
    srand(time(NULL));
    
    // Run all tests
    test_basic_allocation_performance();
    test_vectorized_operations();
    test_concurrent_allocation();
    test_fragmentation_resistance();
    test_prefetch_performance();
    test_memory_stress();
    
    // Print final memory pool statistics
    memory_stats_t final_stats = mev_get_memory_stats();
    printf("\n📊 Final Memory Pool Statistics:\n");
    printf("   Total size:         %zu bytes (%.2f MB)\n", 
           final_stats.total_size, final_stats.total_size / 1024.0 / 1024.0);
    printf("   Used size:          %zu bytes (%.2f MB)\n", 
           final_stats.used_size, final_stats.used_size / 1024.0 / 1024.0);
    printf("   Free size:          %zu bytes (%.2f MB)\n", 
           final_stats.free_size, final_stats.free_size / 1024.0 / 1024.0);
    printf("   Total allocations:  %lu\n", final_stats.allocation_count);
    printf("   Total deallocations: %lu\n", final_stats.deallocation_count);
    printf("   Total allocated:    %lu bytes (%.2f MB)\n", 
           final_stats.total_allocated, final_stats.total_allocated / 1024.0 / 1024.0);
    printf("   Fragmentation ratio: %.4f\n", final_stats.fragmentation_ratio);
    
    // Cleanup
    mev_memory_cleanup();
    
    printf("\n✅ All memory tests passed! Memory management system ready for MEV operations.\n");
    
    return 0;
}
