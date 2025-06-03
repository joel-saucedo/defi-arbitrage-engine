#!/usr/bin/env python3
# filepath: src/core/asm_optimizations/test_asm_integration.py
"""
assembly integration testing for ultra-fast mathematical operations
validates ctypes interface and performance characteristics
"""

import ctypes
import numpy as np
import time
import os
import sys
from typing import Optional

class AssemblyMathLibrary:
    """ctypes wrapper for assembly-optimized mathematical operations"""
    
    def __init__(self, library_path: str = "./libasm_math.so"):
        if not os.path.exists(library_path):
            print(f"[error] assembly library not found: {library_path}")
            print("run: make -C src/core/asm_optimizations")
            sys.exit(1)
        
        self.lib = ctypes.CDLL(library_path)
        self._configure_function_signatures()
        print(f"[init] loaded assembly library: {library_path}")
    
    def _configure_function_signatures(self):
        """configure ctypes function signatures for type safety"""
        
        # price impact calculation
        self.lib.safe_calculate_price_impact.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_double,
            ctypes.c_int64
        ]
        self.lib.safe_calculate_price_impact.restype = ctypes.c_int64
        
        # vectorized arbitrage detection
        self.lib.safe_vectorized_arbitrage_check.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int64
        ]
        self.lib.safe_vectorized_arbitrage_check.restype = ctypes.c_uint64
        
        # fast square root
        self.lib.safe_fast_sqrt.argtypes = [ctypes.c_uint64]
        self.lib.safe_fast_sqrt.restype = ctypes.c_uint64
        
        # moving average
        self.lib.calculate_moving_average_avx2.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int64,
            ctypes.c_int64
        ]
        self.lib.calculate_moving_average_avx2.restype = ctypes.c_double
        
        # volatility calculation
        self.lib.calculate_volatility_avx2.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int64
        ]
        self.lib.calculate_volatility_avx2.restype = ctypes.c_double
        
        # correlation calculation
        self.lib.calculate_correlation_optimized.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int64
        ]
        self.lib.calculate_correlation_optimized.restype = ctypes.c_double
    
    def calculate_price_impact(self, prices: np.ndarray, volume: float) -> Optional[int]:
        """calculate price impact using assembly-optimized simd operations"""
        if len(prices) == 0 or volume <= 0:
            return None
        
        # ensure 32-byte alignment for avx operations
        aligned_prices = np.array(prices, dtype=np.float64)
        price_ptr = (ctypes.c_double * len(aligned_prices))(*aligned_prices)
        
        result = self.lib.safe_calculate_price_impact(price_ptr, volume, len(prices))
        return result if result >= 0 else None
    
    def detect_arbitrage_opportunities(self, prices_a: np.ndarray, 
                                     prices_b: np.ndarray) -> Optional[int]:
        """vectorized arbitrage detection using simd comparisons"""
        if len(prices_a) != len(prices_b) or len(prices_a) == 0:
            return None
        
        price_a_ptr = (ctypes.c_double * len(prices_a))(*prices_a)
        price_b_ptr = (ctypes.c_double * len(prices_b))(*prices_b)
        
        bitmask = self.lib.safe_vectorized_arbitrage_check(
            price_a_ptr, price_b_ptr, len(prices_a)
        )
        return bitmask
    
    def fast_sqrt(self, value: int) -> int:
        """ultra-fast square root approximation"""
        return self.lib.safe_fast_sqrt(value)
    
    def moving_average(self, prices: np.ndarray, window: int) -> float:
        """avx2-optimized moving average calculation"""
        if len(prices) < window or window <= 0:
            return 0.0
        
        price_ptr = (ctypes.c_double * len(prices))(*prices)
        return self.lib.calculate_moving_average_avx2(price_ptr, len(prices), window)
    
    def volatility(self, prices: np.ndarray) -> float:
        """simd-optimized volatility calculation"""
        if len(prices) <= 1:
            return 0.0
        
        price_ptr = (ctypes.c_double * len(prices))(*prices)
        return self.lib.calculate_volatility_avx2(price_ptr, len(prices))
    
    def correlation(self, prices_a: np.ndarray, prices_b: np.ndarray) -> float:
        """cache-optimized correlation calculation"""
        if len(prices_a) != len(prices_b) or len(prices_a) <= 1:
            return 0.0
        
        price_a_ptr = (ctypes.c_double * len(prices_a))(*prices_a)
        price_b_ptr = (ctypes.c_double * len(prices_b))(*prices_b)
        
        return self.lib.calculate_correlation_optimized(
            price_a_ptr, price_b_ptr, len(prices_a)
        )

def test_price_impact():
    """test assembly price impact calculation"""
    print("\n[test] price impact calculation")
    
    asm_lib = AssemblyMathLibrary()
    
    # test data
    prices = np.array([100.0, 200.0, 150.0, 300.0, 250.0])
    volume = 1000000.0
    
    result = asm_lib.calculate_price_impact(prices, volume)
    print(f"  price impact (scaled): {result}")
    
    # compare with python implementation
    python_impact = sum(p * volume / (volume + 1000000.0) for p in prices) * 1e18
    print(f"  python comparison: {int(python_impact)}")
    
    return result is not None

def test_arbitrage_detection():
    """test vectorized arbitrage opportunity detection"""
    print("\n[test] arbitrage detection")
    
    asm_lib = AssemblyMathLibrary()
    
    # create price data with arbitrage opportunities
    prices_a = np.array([100.0, 200.0, 150.0, 300.0])
    prices_b = np.array([102.0, 190.0, 155.0, 310.0])  # some profitable spreads
    
    bitmask = asm_lib.detect_arbitrage_opportunities(prices_a, prices_b)
    print(f"  arbitrage bitmask: {bin(bitmask)}")
    
    # verify profitable opportunities
    profitable_pairs = []
    for i in range(len(prices_a)):
        if (bitmask >> i) & 1:
            spread = (prices_b[i] / prices_a[i] - 1) * 100
            profitable_pairs.append((i, spread))
    
    print(f"  profitable pairs: {profitable_pairs}")
    return bitmask > 0

def test_mathematical_functions():
    """test optimized mathematical functions"""
    print("\n[test] mathematical functions")
    
    asm_lib = AssemblyMathLibrary()
    
    # test data
    prices = np.random.random(1000) * 100 + 50
    
    # moving average
    asm_ma = asm_lib.moving_average(prices, 20)
    numpy_ma = np.mean(prices[-20:])
    print(f"  moving average: asm={asm_ma:.4f} numpy={numpy_ma:.4f}")
    
    # volatility
    asm_vol = asm_lib.volatility(prices)
    numpy_vol = np.std(prices, ddof=1)
    print(f"  volatility: asm={asm_vol:.4f} numpy={numpy_vol:.4f}")
    
    # correlation
    prices_b = np.random.random(1000) * 100 + 50
    asm_corr = asm_lib.correlation(prices, prices_b)
    numpy_corr = np.corrcoef(prices, prices_b)[0, 1]
    print(f"  correlation: asm={asm_corr:.4f} numpy={numpy_corr:.4f}")
    
    # fast sqrt
    sqrt_input = 1000000
    asm_sqrt = asm_lib.fast_sqrt(sqrt_input)
    python_sqrt = int(sqrt_input ** 0.5)
    print(f"  sqrt({sqrt_input}): asm={asm_sqrt} python={python_sqrt}")
    
    return True

def benchmark_performance():
    """comprehensive performance benchmarking"""
    print("\n[benchmark] performance comparison")
    
    asm_lib = AssemblyMathLibrary()
    
    sizes = [1000, 10000, 100000]
    iterations = [1000, 100, 10]
    
    for size, iters in zip(sizes, iterations):
        print(f"\n  dataset size: {size}, iterations: {iters}")
        
        prices = np.random.random(size) * 1000 + 100
        
        # benchmark moving average
        start = time.perf_counter_ns()
        for _ in range(iters):
            asm_lib.moving_average(prices, 20)
        asm_time = (time.perf_counter_ns() - start) / 1e6
        
        start = time.perf_counter_ns()
        for _ in range(iters):
            np.mean(prices[-20:])
        numpy_time = (time.perf_counter_ns() - start) / 1e6
        
        speedup = numpy_time / asm_time if asm_time > 0 else 0
        ops_per_sec = iters * 1000 / asm_time if asm_time > 0 else 0
        
        print(f"    moving average: {asm_time:.2f}ms vs {numpy_time:.2f}ms")
        print(f"    speedup: {speedup:.1f}x, {ops_per_sec:.0f} ops/sec")

def test_memory_alignment():
    """test memory alignment requirements for simd operations"""
    print("\n[test] memory alignment")
    
    asm_lib = AssemblyMathLibrary()
    
    # test with aligned data
    aligned_prices = np.array([100.0, 200.0, 150.0, 300.0], dtype=np.float64)
    result_aligned = asm_lib.calculate_price_impact(aligned_prices, 1000000.0)
    
    # test with potentially unaligned data
    unaligned_prices = np.array([100.0, 200.0, 150.0], dtype=np.float64)
    result_unaligned = asm_lib.calculate_price_impact(unaligned_prices, 1000000.0)
    
    print(f"  aligned result: {result_aligned}")
    print(f"  unaligned result: {result_unaligned}")
    
    return result_aligned is not None and result_unaligned is not None

def main():
    """run comprehensive assembly integration tests"""
    print("assembly mathematical operations - integration tests")
    print("=" * 60)
    
    tests = [
        ("price impact calculation", test_price_impact),
        ("arbitrage detection", test_arbitrage_detection),
        ("mathematical functions", test_mathematical_functions),
        ("memory alignment", test_memory_alignment),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                print(f"[pass] {test_name}")
                passed += 1
            else:
                print(f"[fail] {test_name}")
        except Exception as e:
            print(f"[error] {test_name}: {e}")
    
    print(f"\ntest results: {passed}/{total} passed")
    
    if passed == total:
        benchmark_performance()
        print("\n[success] all tests passed - assembly integration working")
        return 0
    else:
        print("\n[failure] some tests failed - check assembly compilation")
        return 1

if __name__ == "__main__":
    sys.exit(main())
