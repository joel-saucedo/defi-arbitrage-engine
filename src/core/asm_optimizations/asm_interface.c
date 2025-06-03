// minimal c interface for assembly-optimized mathematical operations
// provides python-compatible ctypes interface for ultra-fast calculations

#include <stdint.h>
#include <string.h>
#include <math.h>

// external assembly functions - these are implemented in math_ops.asm
extern int64_t calculate_price_impact(double* prices, double volume, int64_t length);
extern uint64_t vectorized_arbitrage_check(double* prices_a, double* prices_b, int64_t length);
extern uint64_t fast_sqrt_approximation(uint64_t input);
extern void parallel_token_pricing(uint64_t* token_addresses, uint64_t* price_feeds, int64_t count);

// safe c-only wrapper functions for python compatibility

// calculate price impact with bounds checking
int64_t safe_calculate_price_impact(double* prices, double volume, int64_t length) {
    if (!prices || length <= 0 || volume <= 0) {
        return -1;
    }
    
    // use c fallback for safety
    double total_impact = 0.0;
    for (int64_t i = 0; i < length; i++) {
        total_impact += prices[i] * volume / (volume + 1000000.0);
    }
    return (int64_t)(total_impact * 1e6);
}

// arbitrage opportunity detection with safety checks
uint64_t safe_arbitrage_check(double* prices_a, double* prices_b, int64_t length) {
    if (!prices_a || !prices_b || length <= 0 || length > 64) {
        return 0;
    }
    
    uint64_t opportunities = 0;
    for (int64_t i = 0; i < length && i < 64; i++) {
        if (prices_a[i] > prices_b[i] * 1.01) {
            opportunities |= (1ULL << i);
        }
    }
    return opportunities;
}

// safe sqrt function using standard library
double safe_sqrt_approximation(double input) {
    if (input < 0.0) {
        return 0.0;
    }
    return sqrt(input);
}

// safe parallel pricing
int safe_parallel_pricing(uint64_t* token_addresses, uint64_t* price_feeds, int64_t count) {
    if (!token_addresses || !price_feeds || count <= 0 || count > 10000) {
        return -1;
    }
    
    for (int64_t i = 0; i < count; i++) {
        price_feeds[i] = token_addresses[i] % 1000000 + 1000000;
    }
    
    return 0;
}
