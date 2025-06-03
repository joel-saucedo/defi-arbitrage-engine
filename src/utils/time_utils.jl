##################################################
#                ETHEREUM MEV RESEARCH            #
#            TIME UTILITIES - JULIA IMPLEMENTATION #
#               High-Precision Timing Operations   #
##################################################

module TimeUtils

using Dates
using Statistics
using Printf

export TimingContext, PerformanceProfiler, NanosecondTimer
export @time_ns, @profile_block, measure_latency, benchmark_function
export get_system_time_ns, calculate_jitter, moving_average_latency

# High-precision timing context for MEV operations
mutable struct TimingContext
    start_time::UInt64
    end_time::UInt64
    operation_name::String
    measurements::Vector{UInt64}
    
    TimingContext(name::String) = new(0, 0, name, UInt64[])
end

# Performance profiler with statistical analysis
mutable struct PerformanceProfiler
    contexts::Dict{String, TimingContext}
    global_start::UInt64
    total_operations::Int64
    
    PerformanceProfiler() = new(Dict{String, TimingContext}(), time_ns(), 0)
end

# Nanosecond-precision timer
struct NanosecondTimer
    precision::Float64
    overhead::UInt64
    
    function NanosecondTimer()
        # Measure timer overhead
        overhead_measurements = UInt64[]
        for _ in 1:1000
            start = time_ns()
            stop = time_ns()
            push!(overhead_measurements, stop - start)
        end
        avg_overhead = UInt64(mean(overhead_measurements))
        precision = 1.0  # nanosecond precision
        new(precision, avg_overhead)
    end
end

# Global profiler instance
const GLOBAL_PROFILER = PerformanceProfiler()
const GLOBAL_TIMER = NanosecondTimer()

"""
Get system time in nanoseconds with highest available precision
"""
function get_system_time_ns()::UInt64
    return time_ns()
end

"""
Start timing operation with context
"""
function start_timing!(profiler::PerformanceProfiler, operation::String)
    if !haskey(profiler.contexts, operation)
        profiler.contexts[operation] = TimingContext(operation)
    end
    
    context = profiler.contexts[operation]
    context.start_time = get_system_time_ns()
    profiler.total_operations += 1
end

"""
Stop timing operation and record measurement
"""
function stop_timing!(profiler::PerformanceProfiler, operation::String)
    end_time = get_system_time_ns()
    
    if haskey(profiler.contexts, operation)
        context = profiler.contexts[operation]
        context.end_time = end_time
        
        if context.start_time > 0
            elapsed = end_time - context.start_time - GLOBAL_TIMER.overhead
            push!(context.measurements, max(elapsed, 1))  # Minimum 1ns
        end
    end
end

"""
Macro for high-precision timing of code blocks
"""
macro time_ns(expr)
    quote
        local start_time = time_ns()
        local result = $(esc(expr))
        local end_time = time_ns()
        local elapsed = end_time - start_time - GLOBAL_TIMER.overhead
        println(@sprintf("Execution time: %.3f μs (%.0f ns)", 
                elapsed / 1000.0, elapsed))
        result
    end
end

"""
Macro for profiling code blocks with automatic context management
"""
macro profile_block(operation_name, expr)
    quote
        local op_name = $(esc(operation_name))
        start_timing!(GLOBAL_PROFILER, op_name)
        local result = $(esc(expr))
        stop_timing!(GLOBAL_PROFILER, op_name)
        result
    end
end

"""
Measure latency of a function with statistical analysis
"""
function measure_latency(func::Function, args...; iterations::Int=1000)
    measurements = UInt64[]
    
    # Warmup
    for _ in 1:min(100, iterations ÷ 10)
        func(args...)
    end
    
    # Actual measurements
    for _ in 1:iterations
        start_time = time_ns()
        func(args...)
        end_time = time_ns()
        
        elapsed = end_time - start_time - GLOBAL_TIMER.overhead
        push!(measurements, max(elapsed, 1))
    end
    
    return (
        mean = mean(measurements),
        median = median(measurements),
        min = minimum(measurements),
        max = maximum(measurements),
        std = std(measurements),
        p95 = quantile(measurements, 0.95),
        p99 = quantile(measurements, 0.99),
        p999 = quantile(measurements, 0.999),
        measurements = measurements
    )
end

"""
Benchmark function with detailed performance analysis
"""
function benchmark_function(func::Function, args...; 
                          iterations::Int=10000, 
                          warmup::Int=1000,
                          name::String="benchmark")
    
    println("🚀 Benchmarking: $name")
    println("   Iterations: $iterations (warmup: $warmup)")
    
    # Warmup phase
    for _ in 1:warmup
        func(args...)
    end
    
    # Measurement phase
    measurements = UInt64[]
    gc_disable()  # Disable garbage collection during measurement
    
    try
        for i in 1:iterations
            start_time = time_ns()
            result = func(args...)
            end_time = time_ns()
            
            elapsed = end_time - start_time - GLOBAL_TIMER.overhead
            push!(measurements, max(elapsed, 1))
            
            # Progress indicator
            if i % (iterations ÷ 10) == 0
                print(".")
            end
        end
    finally
        gc_enable()  # Re-enable garbage collection
    end
    
    println("\n")
    
    # Statistical analysis
    stats = (
        total_time = sum(measurements),
        mean = mean(measurements),
        median = median(measurements),
        min = minimum(measurements),
        max = maximum(measurements),
        std = std(measurements),
        cv = std(measurements) / mean(measurements),  # Coefficient of variation
        p50 = quantile(measurements, 0.50),
        p90 = quantile(measurements, 0.90),
        p95 = quantile(measurements, 0.95),
        p99 = quantile(measurements, 0.99),
        p999 = quantile(measurements, 0.999),
        throughput = 1_000_000_000 / mean(measurements),  # ops/second
        measurements = measurements
    )
    
    # Print results
    println("📊 Benchmark Results:")
    println(@sprintf("   Total time:    %.2f ms", stats.total_time / 1_000_000))
    println(@sprintf("   Mean latency:  %.3f μs", stats.mean / 1000))
    println(@sprintf("   Median:        %.3f μs", stats.median / 1000))
    println(@sprintf("   Min:           %.3f μs", stats.min / 1000))
    println(@sprintf("   Max:           %.3f μs", stats.max / 1000))
    println(@sprintf("   Std dev:       %.3f μs", stats.std / 1000))
    println(@sprintf("   CV:            %.2f%%", stats.cv * 100))
    println(@sprintf("   P95:           %.3f μs", stats.p95 / 1000))
    println(@sprintf("   P99:           %.3f μs", stats.p99 / 1000))
    println(@sprintf("   P99.9:         %.3f μs", stats.p999 / 1000))
    println(@sprintf("   Throughput:    %.0f ops/sec", stats.throughput))
    
    return stats
end

"""
Calculate timing jitter and stability metrics
"""
function calculate_jitter(measurements::Vector{UInt64})
    if length(measurements) < 2
        return (jitter=0.0, stability=0.0, consistency=0.0)
    end
    
    # Calculate consecutive differences
    diffs = abs.(diff(measurements))
    
    jitter = mean(diffs)
    max_jitter = maximum(diffs)
    stability = 1.0 - (std(measurements) / mean(measurements))
    consistency = 1.0 - (jitter / mean(measurements))
    
    return (
        jitter = jitter,
        max_jitter = max_jitter,
        stability = max(0.0, min(1.0, stability)),
        consistency = max(0.0, min(1.0, consistency))
    )
end

"""
Moving average latency calculator for real-time monitoring
"""
mutable struct MovingAverageLatency
    window_size::Int
    measurements::Vector{UInt64}
    current_average::Float64
    
    MovingAverageLatency(window_size::Int) = new(window_size, UInt64[], 0.0)
end

function add_measurement!(ma::MovingAverageLatency, measurement::UInt64)
    push!(ma.measurements, measurement)
    
    if length(ma.measurements) > ma.window_size
        popfirst!(ma.measurements)
    end
    
    ma.current_average = mean(ma.measurements)
    return ma.current_average
end

"""
Get comprehensive profiler statistics
"""
function get_profiler_stats(profiler::PerformanceProfiler)
    stats = Dict{String, Any}()
    
    stats["total_operations"] = profiler.total_operations
    stats["uptime_ns"] = time_ns() - profiler.global_start
    stats["operations"] = Dict{String, Any}()
    
    for (name, context) in profiler.contexts
        if !isempty(context.measurements)
            op_stats = Dict{String, Any}()
            op_stats["count"] = length(context.measurements)
            op_stats["total_time"] = sum(context.measurements)
            op_stats["mean"] = mean(context.measurements)
            op_stats["median"] = median(context.measurements)
            op_stats["min"] = minimum(context.measurements)
            op_stats["max"] = maximum(context.measurements)
            op_stats["std"] = std(context.measurements)
            op_stats["p95"] = quantile(context.measurements, 0.95)
            op_stats["p99"] = quantile(context.measurements, 0.99)
            
            jitter_stats = calculate_jitter(context.measurements)
            op_stats["jitter"] = jitter_stats
            
            stats["operations"][name] = op_stats
        end
    end
    
    return stats
end

"""
Print formatted profiler report
"""
function print_profiler_report(profiler::PerformanceProfiler = GLOBAL_PROFILER)
    stats = get_profiler_stats(profiler)
    
    println("\n" * "="^60)
    println("           MEV RESEARCH PERFORMANCE REPORT")
    println("="^60)
    println(@sprintf("Total Operations: %d", stats["total_operations"]))
    println(@sprintf("Uptime: %.2f seconds", stats["uptime_ns"] / 1_000_000_000))
    println("\nOperation Statistics:")
    println("-"^60)
    
    for (name, op_stats) in stats["operations"]
        println(@sprintf("\n🔧 %s:", name))
        println(@sprintf("   Count:      %d", op_stats["count"]))
        println(@sprintf("   Mean:       %.3f μs", op_stats["mean"] / 1000))
        println(@sprintf("   Median:     %.3f μs", op_stats["median"] / 1000))
        println(@sprintf("   Min:        %.3f μs", op_stats["min"] / 1000))
        println(@sprintf("   Max:        %.3f μs", op_stats["max"] / 1000))
        println(@sprintf("   P95:        %.3f μs", op_stats["p95"] / 1000))
        println(@sprintf("   P99:        %.3f μs", op_stats["p99"] / 1000))
        println(@sprintf("   Jitter:     %.3f μs", op_stats["jitter"]["jitter"] / 1000))
        println(@sprintf("   Stability:  %.1f%%", op_stats["jitter"]["stability"] * 100))
    end
    
    println("\n" * "="^60)
end

"""
Reset profiler statistics
"""
function reset_profiler!(profiler::PerformanceProfiler = GLOBAL_PROFILER)
    empty!(profiler.contexts)
    profiler.global_start = time_ns()
    profiler.total_operations = 0
end

"""
Export timing data to CSV for analysis
"""
function export_timing_data(profiler::PerformanceProfiler, filename::String)
    open(filename, "w") do file
        println(file, "operation,timestamp,latency_ns")
        
        for (name, context) in profiler.contexts
            for (i, measurement) in enumerate(context.measurements)
                println(file, "$name,$i,$measurement")
            end
        end
    end
    
    println("Timing data exported to: $filename")
end

end # module TimeUtils
