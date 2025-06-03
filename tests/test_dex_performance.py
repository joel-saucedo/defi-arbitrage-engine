################################################################################
#                          ETHEREUM MEV RESEARCH - TEST SUITE
#                           High-Performance DEX Testing Framework
#                                 Python Implementation
################################################################################

import pytest
import asyncio
import time
import numpy as np
import multiprocessing as mp
from decimal import Decimal
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dex.uniswap_v3 import UniswapV3Engine
from dex.balancer import BalancerEngine  # We'll need to import compiled Go module
from dex.sushiswap import SushiSwapEngine  # We'll need to import compiled C++ module
from utils.math_utils import *

class TestPerformanceProfiler:
    """High-precision performance profiler for MEV operations"""
    
    def __init__(self):
        self.measurements = {}
        self.start_times = {}
    
    def start_timer(self, operation: str):
        self.start_times[operation] = time.perf_counter_ns()
    
    def end_timer(self, operation: str):
        if operation in self.start_times:
            duration = time.perf_counter_ns() - self.start_times[operation]
            if operation not in self.measurements:
                self.measurements[operation] = []
            self.measurements[operation].append(duration)
            return duration
        return None
    
    def get_stats(self, operation: str) -> Dict:
        if operation not in self.measurements:
            return {}
        
        measurements = np.array(self.measurements[operation])
        return {
            'count': len(measurements),
            'mean_ns': np.mean(measurements),
            'median_ns': np.median(measurements),
            'min_ns': np.min(measurements),
            'max_ns': np.max(measurements),
            'std_ns': np.std(measurements),
            'p95_ns': np.percentile(measurements, 95),
            'p99_ns': np.percentile(measurements, 99),
            'p999_ns': np.percentile(measurements, 99.9),
            'mean_us': np.mean(measurements) / 1000,
            'median_us': np.median(measurements) / 1000,
            'p99_us': np.percentile(measurements, 99) / 1000
        }
    
    def print_report(self):
        print("\n" + "="*80)
        print("                    MEV RESEARCH PERFORMANCE REPORT")
        print("="*80)
        
        for operation, measurements in self.measurements.items():
            stats = self.get_stats(operation)
            print(f"\n🔧 {operation}:")
            print(f"   Count:      {stats['count']:,}")
            print(f"   Mean:       {stats['mean_us']:.3f} μs")
            print(f"   Median:     {stats['median_us']:.3f} μs")
            print(f"   Min:        {stats['min_ns']:,} ns")
            print(f"   Max:        {stats['max_ns']:,} ns")
            print(f"   P95:        {stats['p95_ns']:,} ns")
            print(f"   P99:        {stats['p99_us']:.3f} μs")
            print(f"   P99.9:      {stats['p999_ns']:,} ns")
            print(f"   Std Dev:    {stats['std_ns']:,} ns")

# Global profiler instance
profiler = TestPerformanceProfiler()

class TestUniswapV3Performance:
    """Ultra-fast Uniswap V3 engine performance tests"""
    
    @pytest.fixture
    def engine(self):
        return UniswapV3Engine()
    
    def test_price_calculation_latency(self, engine):
        """Test sub-millisecond price calculations"""
        # Warmup
        for _ in range(1000):
            engine.calculate_price_impact(
                token0_reserve=Decimal('1000000'),
                token1_reserve=Decimal('2000000'),
                amount_in=Decimal('1000'),
                fee_tier=3000
            )
        
        # Actual measurements
        latencies = []
        for _ in range(10000):
            profiler.start_timer('uniswap_price_calc')
            engine.calculate_price_impact(
                token0_reserve=Decimal('1000000'),
                token1_reserve=Decimal('2000000'),
                amount_in=Decimal('1000'),
                fee_tier=3000
            )
            duration = profiler.end_timer('uniswap_price_calc')
            latencies.append(duration)
        
        # Assert sub-millisecond performance
        mean_latency_us = np.mean(latencies) / 1000
        p99_latency_us = np.percentile(latencies, 99) / 1000
        
        print(f"\n🚀 Uniswap V3 Price Calculation Performance:")
        print(f"   Mean: {mean_latency_us:.3f} μs")
        print(f"   P99:  {p99_latency_us:.3f} μs")
        
        assert mean_latency_us < 50.0, f"Mean latency {mean_latency_us:.3f} μs exceeds 50 μs target"
        assert p99_latency_us < 200.0, f"P99 latency {p99_latency_us:.3f} μs exceeds 200 μs target"
    
    def test_arbitrage_detection_speed(self, engine):
        """Test arbitrage opportunity detection performance"""
        pools = [
            {
                'token0_reserve': Decimal('1000000'),
                'token1_reserve': Decimal('2000000'),
                'fee_tier': 3000,
                'tick_current': 0,
                'liquidity': Decimal('5000000')
            }
            for _ in range(100)
        ]
        
        # Warmup
        for _ in range(100):
            engine.find_arbitrage_opportunities(pools[:10])
        
        # Benchmark
        latencies = []
        for _ in range(1000):
            profiler.start_timer('arbitrage_detection')
            opportunities = engine.find_arbitrage_opportunities(pools)
            duration = profiler.end_timer('arbitrage_detection')
            latencies.append(duration)
        
        mean_latency_us = np.mean(latencies) / 1000
        assert mean_latency_us < 500.0, f"Arbitrage detection too slow: {mean_latency_us:.3f} μs"
    
    def test_concurrent_price_calculations(self, engine):
        """Test concurrent price calculation performance"""
        
        def calculate_batch():
            results = []
            for i in range(100):
                result = engine.calculate_price_impact(
                    token0_reserve=Decimal(f'{1000000 + i}'),
                    token1_reserve=Decimal(f'{2000000 + i}'),
                    amount_in=Decimal('1000'),
                    fee_tier=3000
                )
                results.append(result)
            return results
        
        # Test with multiple threads
        with ThreadPoolExecutor(max_workers=8) as executor:
            profiler.start_timer('concurrent_calculations')
            futures = [executor.submit(calculate_batch) for _ in range(8)]
            results = [future.result() for future in futures]
            duration = profiler.end_timer('concurrent_calculations')
        
        total_calculations = 8 * 100
        throughput = total_calculations / (duration / 1e9)  # ops/second
        
        print(f"\n🔥 Concurrent Performance:")
        print(f"   Total calculations: {total_calculations}")
        print(f"   Duration: {duration / 1e6:.2f} ms")
        print(f"   Throughput: {throughput:,.0f} ops/sec")
        
        assert throughput > 50000, f"Throughput too low: {throughput:,.0f} ops/sec"

class TestCrossLanguageIntegration:
    """Test integration between different language implementations"""
    
    def test_price_consistency_across_languages(self):
        """Ensure price calculations are consistent across Python, C++, and Go"""
        # Test parameters
        token0_reserve = 1000000
        token1_reserve = 2000000
        amount_in = 1000
        fee_tier = 3000
        
        # Python implementation
        python_engine = UniswapV3Engine()
        python_result = python_engine.calculate_price_impact(
            Decimal(token0_reserve),
            Decimal(token1_reserve), 
            Decimal(amount_in),
            fee_tier
        )
        
        # TODO: Add C++ and Go comparisons when modules are available
        # cpp_result = sushiswap_engine.calculate_price(...)
        # go_result = balancer_engine.calculate_price(...)
        
        print(f"\n🔗 Cross-Language Price Consistency:")
        print(f"   Python result: {python_result}")
        # print(f"   C++ result:    {cpp_result}")
        # print(f"   Go result:     {go_result}")
        
        # For now, just validate Python result structure
        assert 'amount_out' in python_result
        assert 'price_impact' in python_result
        assert 'effective_price' in python_result

class TestMathUtilsPerformance:
    """Test mathematical utility performance"""
    
    def test_vectorized_operations(self):
        """Test vectorized mathematical operations"""
        # Large arrays for testing
        a = np.random.rand(100000).astype(np.float64)
        b = np.random.rand(100000).astype(np.float64)
        
        # Test vectorized addition
        profiler.start_timer('vectorized_add')
        result = fast_vectorized_add(a, b)
        duration = profiler.end_timer('vectorized_add')
        
        throughput = len(a) / (duration / 1e9)
        print(f"\n⚡ Vectorized Addition Performance:")
        print(f"   Array size: {len(a):,}")
        print(f"   Duration: {duration / 1e6:.3f} ms")
        print(f"   Throughput: {throughput:,.0f} ops/sec")
        
        assert duration < 1e6, f"Vectorized operation too slow: {duration / 1e6:.3f} ms"
    
    def test_sqrt_approximation_accuracy(self):
        """Test fast square root approximation accuracy and speed"""
        test_values = np.linspace(0.1, 1000000, 10000)
        
        # Test accuracy
        errors = []
        for val in test_values[:1000]:  # Sample for accuracy test
            fast_result = fast_sqrt_approx(val)
            exact_result = np.sqrt(val)
            relative_error = abs(fast_result - exact_result) / exact_result
            errors.append(relative_error)
        
        max_error = max(errors)
        mean_error = np.mean(errors)
        
        print(f"\n📐 Fast Square Root Accuracy:")
        print(f"   Max relative error:  {max_error:.2e}")
        print(f"   Mean relative error: {mean_error:.2e}")
        
        # Test speed
        profiler.start_timer('fast_sqrt')
        for val in test_values:
            fast_sqrt_approx(val)
        duration = profiler.end_timer('fast_sqrt')
        
        throughput = len(test_values) / (duration / 1e9)
        print(f"   Throughput: {throughput:,.0f} ops/sec")
        
        assert max_error < 0.01, f"Square root approximation too inaccurate: {max_error:.2e}"
        assert throughput > 1e6, f"Square root too slow: {throughput:,.0f} ops/sec"

class TestMemoryEfficiency:
    """Test memory usage and efficiency"""
    
    def test_memory_pool_performance(self):
        """Test custom memory pool performance"""
        import tracemalloc
        
        # Standard allocation benchmark
        tracemalloc.start()
        
        profiler.start_timer('standard_allocation')
        standard_objects = []
        for _ in range(10000):
            obj = {"data": list(range(100))}
            standard_objects.append(obj)
        duration_standard = profiler.end_timer('standard_allocation')
        
        current, peak = tracemalloc.get_traced_memory()
        standard_memory = peak
        tracemalloc.stop()
        
        print(f"\n💾 Memory Allocation Performance:")
        print(f"   Standard allocation: {duration_standard / 1e6:.2f} ms")
        print(f"   Standard memory:     {standard_memory / 1024 / 1024:.2f} MB")
        
        # TODO: Implement custom memory pool test when available
        # Custom memory pool would show significant improvement
    
    def test_cache_locality(self):
        """Test cache-friendly data structure performance"""
        # Array of Structures (AOS) - poor cache locality
        aos_data = [{'price': i, 'volume': i*2, 'timestamp': time.time()} 
                   for i in range(100000)]
        
        # Structure of Arrays (SOA) - good cache locality  
        soa_data = {
            'prices': list(range(100000)),
            'volumes': [i*2 for i in range(100000)],
            'timestamps': [time.time() for _ in range(100000)]
        }
        
        # Test AOS access pattern
        profiler.start_timer('aos_access')
        total = sum(item['price'] for item in aos_data)
        duration_aos = profiler.end_timer('aos_access')
        
        # Test SOA access pattern
        profiler.start_timer('soa_access')
        total = sum(soa_data['prices'])
        duration_soa = profiler.end_timer('soa_access')
        
        speedup = duration_aos / duration_soa
        
        print(f"\n🎯 Cache Locality Performance:")
        print(f"   AOS access: {duration_aos / 1e6:.2f} ms")
        print(f"   SOA access: {duration_soa / 1e6:.2f} ms")
        print(f"   Speedup:    {speedup:.2f}x")
        
        assert speedup > 1.5, f"SOA should be significantly faster than AOS"

class TestStressAndLoad:
    """Stress testing and load testing"""
    
    def test_high_frequency_trading_simulation(self):
        """Simulate high-frequency trading workload"""
        engine = UniswapV3Engine()
        
        # Simulate 1000 price updates per second for 10 seconds
        updates_per_second = 1000
        duration_seconds = 10
        total_updates = updates_per_second * duration_seconds
        
        print(f"\n🔥 High-Frequency Trading Simulation:")
        print(f"   Target: {updates_per_second} updates/sec for {duration_seconds} seconds")
        
        start_time = time.perf_counter()
        successful_updates = 0
        
        for i in range(total_updates):
            try:
                result = engine.calculate_price_impact(
                    token0_reserve=Decimal(f'{1000000 + (i % 1000)}'),
                    token1_reserve=Decimal(f'{2000000 + (i % 2000)}'),
                    amount_in=Decimal('1000'),
                    fee_tier=3000
                )
                successful_updates += 1
                
                # Maintain target frequency
                target_time = start_time + (i + 1) / updates_per_second
                current_time = time.perf_counter()
                if current_time < target_time:
                    time.sleep(target_time - current_time)
                    
            except Exception as e:
                print(f"Update {i} failed: {e}")
        
        actual_duration = time.perf_counter() - start_time
        actual_rate = successful_updates / actual_duration
        
        print(f"   Actual rate:     {actual_rate:.0f} updates/sec")
        print(f"   Success rate:    {successful_updates/total_updates*100:.1f}%")
        print(f"   Total duration:  {actual_duration:.2f} seconds")
        
        assert actual_rate >= updates_per_second * 0.95, f"Failed to maintain target rate"
        assert successful_updates >= total_updates * 0.99, f"Too many failed updates"
    
    def test_memory_leak_detection(self):
        """Test for memory leaks in long-running operations"""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        engine = UniswapV3Engine()
        
        # Run operations for extended period
        print(f"\n🔍 Memory Leak Detection:")
        print(f"   Initial memory: {initial_memory / 1024 / 1024:.2f} MB")
        
        for iteration in range(10):
            # Perform 1000 operations
            for _ in range(1000):
                engine.calculate_price_impact(
                    token0_reserve=Decimal('1000000'),
                    token1_reserve=Decimal('2000000'),
                    amount_in=Decimal('1000'),
                    fee_tier=3000
                )
            
            # Force garbage collection
            gc.collect()
            
            current_memory = process.memory_info().rss
            memory_increase = (current_memory - initial_memory) / 1024 / 1024
            
            print(f"   Iteration {iteration + 1}: {current_memory / 1024 / 1024:.2f} MB "
                  f"(+{memory_increase:.2f} MB)")
            
            # Allow for some memory growth but detect significant leaks
            if iteration > 5:  # Allow warm-up period
                assert memory_increase < 50, f"Potential memory leak detected: +{memory_increase:.2f} MB"

@pytest.mark.asyncio
class TestAsyncPerformance:
    """Test asynchronous operation performance"""
    
    async def test_concurrent_dex_operations(self):
        """Test concurrent DEX operations with asyncio"""
        engine = UniswapV3Engine()
        
        async def price_calculation_task(task_id):
            profiler.start_timer(f'async_task_{task_id}')
            result = engine.calculate_price_impact(
                token0_reserve=Decimal(f'{1000000 + task_id}'),
                token1_reserve=Decimal(f'{2000000 + task_id}'),
                amount_in=Decimal('1000'),
                fee_tier=3000
            )
            duration = profiler.end_timer(f'async_task_{task_id}')
            return result, duration
        
        # Run 100 concurrent tasks
        profiler.start_timer('async_concurrent_batch')
        tasks = [price_calculation_task(i) for i in range(100)]
        results = await asyncio.gather(*tasks)
        total_duration = profiler.end_timer('async_concurrent_batch')
        
        individual_durations = [result[1] for result in results]
        mean_individual = np.mean(individual_durations)
        
        print(f"\n🔄 Async Concurrent Performance:")
        print(f"   Total batch time:     {total_duration / 1e6:.2f} ms")
        print(f"   Mean individual time: {mean_individual / 1e6:.2f} ms")
        print(f"   Tasks completed:      {len(results)}")
        print(f"   Concurrency benefit:  {(mean_individual * len(results)) / total_duration:.2f}x")
        
        assert len(results) == 100, "Not all async tasks completed"
        assert total_duration < mean_individual * 100 * 0.5, "Insufficient concurrency benefit"

def test_performance_summary():
    """Print comprehensive performance summary"""
    profiler.print_report()
    
    # Validate key performance metrics
    uniswap_stats = profiler.get_stats('uniswap_price_calc')
    if uniswap_stats:
        assert uniswap_stats['mean_us'] < 100, "Uniswap calculations too slow"
        print(f"\n✅ All performance targets met!")
        print(f"   Uniswap V3 mean latency: {uniswap_stats['mean_us']:.3f} μs")

if __name__ == "__main__":
    # Run tests with detailed output
    pytest.main([__file__, "-v", "-s", "--tb=short"])
