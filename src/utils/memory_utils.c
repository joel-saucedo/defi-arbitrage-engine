/*
 * ═══════════════════════════════════════════════════════════════════════════════════
 *                         ETHEREUM MEV RESEARCH - MEMORY UTILITIES
 *                              High-Performance Memory Management
 *                                       C Implementation
 * ═══════════════════════════════════════════════════════════════════════════════════
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/mman.h>
#include <errno.h>
#include <immintrin.h>
#include <pthread.h>

#ifndef MEMORY_UTILS_H
#define MEMORY_UTILS_H

// Memory pool configuration
#define MEMORY_POOL_SIZE (1024 * 1024 * 1024)  // 1GB pool
#define BLOCK_SIZES_COUNT 8
#define MAX_BLOCKS_PER_SIZE 10000
#define CACHE_LINE_SIZE 64
#define PAGE_SIZE 4096

// Memory block structure with cache-aligned headers
typedef struct memory_block {
    size_t size;
    uint8_t is_free;
    uint8_t pool_index;
    uint16_t magic;  // Corruption detection
    struct memory_block* next;
    struct memory_block* prev;
    uint8_t padding[CACHE_LINE_SIZE - sizeof(size_t) - 2 - sizeof(uint16_t) - 2 * sizeof(void*)];
} __attribute__((aligned(CACHE_LINE_SIZE))) memory_block_t;

// High-performance memory pool
typedef struct memory_pool {
    void* base_address;
    size_t total_size;
    size_t used_size;
    memory_block_t* free_lists[BLOCK_SIZES_COUNT];
    size_t block_sizes[BLOCK_SIZES_COUNT];
    pthread_mutex_t mutex;
    uint64_t allocation_count;
    uint64_t deallocation_count;
    uint64_t total_allocated;
} memory_pool_t;

// Global memory pool instance
static memory_pool_t* global_pool = NULL;
static pthread_once_t pool_init_once = PTHREAD_ONCE_INIT;

// Standard block sizes for efficient allocation
static const size_t STANDARD_BLOCK_SIZES[BLOCK_SIZES_COUNT] = {
    64, 128, 256, 512, 1024, 2048, 4096, 8192
};

/*
 * Initialize the global memory pool with huge pages support
 */
static void init_memory_pool() {
    global_pool = (memory_pool_t*)malloc(sizeof(memory_pool_t));
    if (!global_pool) {
        fprintf(stderr, "Failed to allocate memory pool structure\n");
        exit(1);
    }

    // Try to allocate with huge pages for better TLB performance
    global_pool->base_address = mmap(NULL, MEMORY_POOL_SIZE, 
                                   PROT_READ | PROT_WRITE,
                                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                                   -1, 0);
    
    if (global_pool->base_address == MAP_FAILED) {
        // Fallback to regular pages
        global_pool->base_address = mmap(NULL, MEMORY_POOL_SIZE,
                                       PROT_READ | PROT_WRITE,
                                       MAP_PRIVATE | MAP_ANONYMOUS,
                                       -1, 0);
        if (global_pool->base_address == MAP_FAILED) {
            fprintf(stderr, "Failed to allocate memory pool: %s\n", strerror(errno));
            exit(1);
        }
    }

    global_pool->total_size = MEMORY_POOL_SIZE;
    global_pool->used_size = 0;
    global_pool->allocation_count = 0;
    global_pool->deallocation_count = 0;
    global_pool->total_allocated = 0;

    // Initialize block sizes and free lists
    for (int i = 0; i < BLOCK_SIZES_COUNT; i++) {
        global_pool->block_sizes[i] = STANDARD_BLOCK_SIZES[i];
        global_pool->free_lists[i] = NULL;
    }

    pthread_mutex_init(&global_pool->mutex, NULL);
    
    // Pre-allocate blocks for common sizes
    for (int i = 0; i < BLOCK_SIZES_COUNT; i++) {
        for (int j = 0; j < 100; j++) {
    pthread_once(&pool_init_once, init_memory_pool);
    
    if (size == 0) return NULL;
    
    // Align size to cache line boundary
    size = (size + CACHE_LINE_SIZE - 1) & ~(CACHE_LINE_SIZE - 1);
    
    int size_class = get_size_class(size);
    
    pthread_mutex_lock(&global_pool->mutex);
    
    memory_block_t* block = NULL;
    
    if (size_class >= 0) {
        // Try to get from free list first
        if (global_pool->free_lists[size_class] != NULL) {
            block = global_pool->free_lists[size_class];
            global_pool->free_lists[size_class] = block->next;
            if (block->next) {
                block->next->prev = NULL;
            }
            block->is_free = 0;
            block->magic = 0xDEAD;
        }
    }
    
    if (!block) {
        // Allocate new block from pool
        size_t block_size = (size_class >= 0) ? 
                           global_pool->block_sizes[size_class] : size;
        size_t total_size = sizeof(memory_block_t) + block_size;
        
        if (global_pool->used_size + total_size > global_pool->total_size) {
            pthread_mutex_unlock(&global_pool->mutex);
            return NULL; // Pool exhausted
        }
        
        block = (memory_block_t*)((uint8_t*)global_pool->base_address + global_pool->used_size);
        block->size = block_size;
        block->is_free = 0;
        block->pool_index = (size_class >= 0) ? size_class : 255;
        block->magic = 0xDEAD;
        block->next = NULL;
        block->prev = NULL;
        
        global_pool->used_size += total_size;
    }
    
    global_pool->allocation_count++;
    global_pool->total_allocated += block->size;
    
    pthread_mutex_unlock(&global_pool->mutex);
    
    // Return pointer after header
    return (uint8_t*)block + sizeof(memory_block_t);
}

/*
 * High-performance memory deallocation with immediate reuse
 */
void mev_free(void* ptr) {
    if (!ptr || !global_pool) return;
    
    memory_block_t* block = (memory_block_t*)((uint8_t*)ptr - sizeof(memory_block_t));
    
    // Verify block integrity
    if (block->magic != 0xDEAD) {
        fprintf(stderr, "Memory corruption detected in block %p\n", ptr);
        return;
    }
    
    pthread_mutex_lock(&global_pool->mutex);
    
    block->is_free = 1;
    block->magic = 0xBEEF;
    
    // Add to appropriate free list
    if (block->pool_index < BLOCK_SIZES_COUNT) {
        block->next = global_pool->free_lists[block->pool_index];
        block->prev = NULL;
        if (block->next) {
            block->next->prev = block;
        }
        global_pool->free_lists[block->pool_index] = block;
    }
    
    global_pool->deallocation_count++;
    
    pthread_mutex_unlock(&global_pool->mutex);
}

/*
 * Vectorized memory copy using AVX2 instructions
 */
void* mev_memcpy(void* dest, const void* src, size_t n) {
    uint8_t* d = (uint8_t*)dest;
    const uint8_t* s = (const uint8_t*)src;
    
    // Handle small copies with regular memcpy
    if (n < 32) {
        return memcpy(dest, src, n);
    }
    
    // AVX2 vectorized copy for large blocks
    size_t avx_chunks = n / 32;
    size_t remaining = n % 32;
    
    for (size_t i = 0; i < avx_chunks; i++) {
        __m256i data = _mm256_loadu_si256((__m256i*)(s + i * 32));
        _mm256_storeu_si256((__m256i*)(d + i * 32), data);
    }
    
    // Handle remaining bytes
    if (remaining > 0) {
        memcpy(d + avx_chunks * 32, s + avx_chunks * 32, remaining);
    }
    
    return dest;
}

/*
 * Vectorized memory set using AVX2 instructions
 */
void* mev_memset(void* ptr, int value, size_t n) {
    uint8_t* p = (uint8_t*)ptr;
    uint8_t val = (uint8_t)value;
    
    // Handle small sets with regular memset
    if (n < 32) {
        return memset(ptr, value, n);
    }
    
    // Create 32-byte pattern
    __m256i pattern = _mm256_set1_epi8(val);
    
    size_t avx_chunks = n / 32;
    size_t remaining = n % 32;
    
    for (size_t i = 0; i < avx_chunks; i++) {
        _mm256_storeu_si256((__m256i*)(p + i * 32), pattern);
    }
    
    // Handle remaining bytes
    if (remaining > 0) {
        memset(p + avx_chunks * 32, value, remaining);
    }
    
    return ptr;
}

/*
 * Prefetch memory for improved cache performance
 */
void mev_prefetch(const void* addr, int locality) {
    // locality: 0 = no temporal locality, 3 = high temporal locality
    switch (locality) {
        case 0:
            _mm_prefetch((const char*)addr, _MM_HINT_NTA);
            break;
        case 1:
            _mm_prefetch((const char*)addr, _MM_HINT_T2);
            break;
        case 2:
            _mm_prefetch((const char*)addr, _MM_HINT_T1);
            break;
        case 3:
        default:
            _mm_prefetch((const char*)addr, _MM_HINT_T0);
            break;
    }
}

/*
 * Batch prefetch for multiple addresses
 */
void mev_batch_prefetch(const void** addresses, size_t count, int locality) {
    for (size_t i = 0; i < count; i++) {
        mev_prefetch(addresses[i], locality);
    }
}

/*
 * Memory barrier operations for synchronization
 */
void mev_memory_barrier() {
    _mm_mfence();
}

void mev_read_barrier() {
    _mm_lfence();
}

void mev_write_barrier() {
    _mm_sfence();
}

/*
 * Get memory pool statistics
 */
typedef struct {
    size_t total_size;
    size_t used_size;
    size_t free_size;
    uint64_t allocation_count;
    uint64_t deallocation_count;
    uint64_t total_allocated;
    double fragmentation_ratio;
} memory_stats_t;

memory_stats_t mev_get_memory_stats() {
    memory_stats_t stats = {0};
    
    if (!global_pool) {
        return stats;
    }
    
    pthread_mutex_lock(&global_pool->mutex);
    
    stats.total_size = global_pool->total_size;
    stats.used_size = global_pool->used_size;
    stats.free_size = global_pool->total_size - global_pool->used_size;
    stats.allocation_count = global_pool->allocation_count;
    stats.deallocation_count = global_pool->deallocation_count;
    stats.total_allocated = global_pool->total_allocated;
    
    // Calculate fragmentation ratio
    size_t free_blocks = 0;
    for (int i = 0; i < BLOCK_SIZES_COUNT; i++) {
        memory_block_t* block = global_pool->free_lists[i];
        while (block) {
            free_blocks++;
            block = block->next;
        }
    }
    
    if (stats.free_size > 0) {
        stats.fragmentation_ratio = (double)free_blocks / (stats.free_size / 1024);
    }
    
    pthread_mutex_unlock(&global_pool->mutex);
    
    return stats;
}

/*
 * Cleanup memory pool on exit
 */
void mev_memory_cleanup() {
    if (global_pool) {
        pthread_mutex_destroy(&global_pool->mutex);
        munmap(global_pool->base_address, global_pool->total_size);
        free(global_pool);
        global_pool = NULL;
    }
}

#endif // MEMORY_UTILS_H
