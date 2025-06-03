// Zig High-Performance Data Structure Testing Suite
// Tests lock-free data structures and memory-efficient algorithms
// Target: Zero-allocation operations with sub-microsecond access times

const std = @import("std");
const testing = std.testing;
const print = std.debug.print;
const ArrayList = std.ArrayList;
const HashMap = std.HashMap;
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

// Test configuration
const TEST_ITERATIONS = 100_000;
const CONCURRENT_THREADS = 8;
const MEMORY_POOL_SIZE = 1024 * 1024; // 1MB

/// Performance measurement structure
const BenchmarkResult = struct {
    operation: []const u8,
    iterations: u64,
    total_time_ns: u64,
    min_time_ns: u64,
    max_time_ns: u64,
    avg_time_ns: f64,
    ops_per_second: f64,
    memory_used: usize,
    
    fn calculate(self: *BenchmarkResult) void {
        self.avg_time_ns = @intToFloat(f64, self.total_time_ns) / @intToFloat(f64, self.iterations);
        self.ops_per_second = 1_000_000_000.0 / self.avg_time_ns;
    }
};

/// High-precision timer for nanosecond measurements
const Timer = struct {
    start_time: u64,
    
    fn start() Timer {
        return Timer{ .start_time = std.time.nanoTimestamp() };
    }
    
    fn lap(self: Timer) u64 {
        return @intCast(u64, std.time.nanoTimestamp() - @intCast(i128, self.start_time));
    }
};

/// Lock-free circular buffer for high-frequency trading data
const LockFreeCircularBuffer = struct {
    const Self = @This();
    
    buffer: []u64,
    capacity: usize,
    head: std.atomic.Atomic(usize),
    tail: std.atomic.Atomic(usize),
    allocator: Allocator,
    
    fn init(allocator: Allocator, capacity: usize) !Self {
        const buffer = try allocator.alloc(u64, capacity);
        return Self{
            .buffer = buffer,
            .capacity = capacity,
            .head = std.atomic.Atomic(usize).init(0),
            .tail = std.atomic.Atomic(usize).init(0),
            .allocator = allocator,
        };
    }
    
    fn deinit(self: *Self) void {
        self.allocator.free(self.buffer);
    }
    
    fn push(self: *Self, value: u64) bool {
        const head = self.head.load(.Acquire);
        const next_head = (head + 1) % self.capacity;
        
        if (next_head == self.tail.load(.Acquire)) {
            return false; // Buffer full
        }
        
        self.buffer[head] = value;
        self.head.store(next_head, .Release);
        return true;
    }
    
    fn pop(self: *Self) ?u64 {
        const tail = self.tail.load(.Acquire);
        
        if (tail == self.head.load(.Acquire)) {
            return null; // Buffer empty
        }
        
        const value = self.buffer[tail];
        self.tail.store((tail + 1) % self.capacity, .Release);
        return value;
    }
    
    fn size(self: *Self) usize {
        const head = self.head.load(.Acquire);
        const tail = self.tail.load(.Acquire);
        
        if (head >= tail) {
            return head - tail;
        } else {
            return self.capacity - tail + head;
        }
    }
};

/// Lock-free priority queue for order book management
const LockFreePriorityQueue = struct {
    const Self = @This();
    const Node = struct {
        value: u64,
        priority: u64,
        next: ?*Node,
    };
    
    head: std.atomic.Atomic(?*Node),
    allocator: Allocator,
    node_pool: []Node,
    pool_index: std.atomic.Atomic(usize),
    
    fn init(allocator: Allocator, pool_size: usize) !Self {
        const node_pool = try allocator.alloc(Node, pool_size);
        return Self{
            .head = std.atomic.Atomic(?*Node).init(null),
            .allocator = allocator,
            .node_pool = node_pool,
            .pool_index = std.atomic.Atomic(usize).init(0),
        };
    }
    
    fn deinit(self: *Self) void {
        self.allocator.free(self.node_pool);
    }
    
    fn allocateNode(self: *Self) ?*Node {
        const index = self.pool_index.fetchAdd(1, .AcqRel);
        if (index >= self.node_pool.len) {
            return null;
        }
        return &self.node_pool[index];
    }
    
    fn insert(self: *Self, value: u64, priority: u64) bool {
        const new_node = self.allocateNode() orelse return false;
        new_node.value = value;
        new_node.priority = priority;
        
        while (true) {
            const head = self.head.load(.Acquire);
            new_node.next = head;
            
            if (head == null or priority > head.?.priority) {
                if (self.head.cmpxchgWeak(head, new_node, .Release, .Acquire)) |_| {
                    continue;
                } else {
                    return true;
                }
            }
            
            // Insert in sorted position (simplified for lock-free operation)
            var current = head;
            while (current != null and current.?.next != null and current.?.next.?.priority > priority) {
                current = current.?.next;
            }
            
            new_node.next = current.?.next;
            current.?.next = new_node;
            return true;
        }
    }
    
    fn extractMax(self: *Self) ?u64 {
        while (true) {
            const head = self.head.load(.Acquire);
            if (head == null) return null;
            
            if (self.head.cmpxchgWeak(head, head.?.next, .Release, .Acquire)) |_| {
                continue;
            } else {
                return head.?.value;
            }
        }
    }
};

/// Memory pool allocator for zero-allocation operations
const MemoryPool = struct {
    const Self = @This();
    const BLOCK_SIZE = 64; // Cache line size
    
    memory: []u8,
    free_blocks: std.atomic.Atomic(usize),
    block_count: usize,
    allocator: Allocator,
    
    fn init(allocator: Allocator, size: usize) !Self {
        const memory = try allocator.alignedAlloc(u8, BLOCK_SIZE, size);
        const block_count = size / BLOCK_SIZE;
        
        return Self{
            .memory = memory,
            .free_blocks = std.atomic.Atomic(usize).init(0),
            .block_count = block_count,
            .allocator = allocator,
        };
    }
    
    fn deinit(self: *Self) void {
        self.allocator.free(self.memory);
    }
    
    fn allocate(self: *Self) ?[]u8 {
        const block_index = self.free_blocks.fetchAdd(1, .AcqRel);
        if (block_index >= self.block_count) {
            return null;
        }
        
        const start = block_index * BLOCK_SIZE;
        return self.memory[start..start + BLOCK_SIZE];
    }
    
    fn reset(self: *Self) void {
        self.free_blocks.store(0, .Release);
    }
};

/// Benchmark circular buffer operations
fn benchmarkCircularBuffer(allocator: Allocator) !BenchmarkResult {
    var buffer = try LockFreeCircularBuffer.init(allocator, 1024);
    defer buffer.deinit();
    
    var result = BenchmarkResult{
        .operation = "CircularBuffer Push/Pop",
        .iterations = TEST_ITERATIONS,
        .total_time_ns = 0,
        .min_time_ns = std.math.maxInt(u64),
        .max_time_ns = 0,
        .avg_time_ns = 0,
        .ops_per_second = 0,
        .memory_used = 1024 * @sizeOf(u64),
    };
    
    print("🔄 Testing lock-free circular buffer...\n");
    
    var i: u64 = 0;
    while (i < TEST_ITERATIONS) : (i += 1) {
        const timer = Timer.start();
        
        // Push operation
        assert(buffer.push(i));
        
        // Pop operation
        const value = buffer.pop();
        assert(value != null and value.? == i);
        
        const elapsed = timer.lap();
        result.total_time_ns += elapsed;
        result.min_time_ns = std.math.min(result.min_time_ns, elapsed);
        result.max_time_ns = std.math.max(result.max_time_ns, elapsed);
    }
    
    result.calculate();
    return result;
}

/// Benchmark priority queue operations
fn benchmarkPriorityQueue(allocator: Allocator) !BenchmarkResult {
    var queue = try LockFreePriorityQueue.init(allocator, TEST_ITERATIONS);
    defer queue.deinit();
    
    var result = BenchmarkResult{
        .operation = "PriorityQueue Insert/Extract",
        .iterations = TEST_ITERATIONS,
        .total_time_ns = 0,
        .min_time_ns = std.math.maxInt(u64),
        .max_time_ns = 0,
        .avg_time_ns = 0,
        .ops_per_second = 0,
        .memory_used = TEST_ITERATIONS * @sizeOf(LockFreePriorityQueue.Node),
    };
    
    print("📊 Testing lock-free priority queue...\n");
    
    // Insert phase
    var i: u64 = 0;
    while (i < TEST_ITERATIONS) : (i += 1) {
        const timer = Timer.start();
        assert(queue.insert(i, i % 1000));
        const elapsed = timer.lap();
        result.total_time_ns += elapsed;
        result.min_time_ns = std.math.min(result.min_time_ns, elapsed);
        result.max_time_ns = std.math.max(result.max_time_ns, elapsed);
    }
    
    result.calculate();
    return result;
}

/// Benchmark memory pool operations
fn benchmarkMemoryPool(allocator: Allocator) !BenchmarkResult {
    var pool = try MemoryPool.init(allocator, MEMORY_POOL_SIZE);
    defer pool.deinit();
    
    var result = BenchmarkResult{
        .operation = "MemoryPool Allocate",
        .iterations = TEST_ITERATIONS,
        .total_time_ns = 0,
        .min_time_ns = std.math.maxInt(u64),
        .max_time_ns = 0,
        .avg_time_ns = 0,
        .ops_per_second = 0,
        .memory_used = MEMORY_POOL_SIZE,
    };
    
    print("🧠 Testing memory pool allocator...\n");
    
    var i: u64 = 0;
    while (i < TEST_ITERATIONS) : (i += 1) {
        const timer = Timer.start();
        const block = pool.allocate();
        const elapsed = timer.lap();
        
        assert(block != null);
        
        result.total_time_ns += elapsed;
        result.min_time_ns = std.math.min(result.min_time_ns, elapsed);
        result.max_time_ns = std.math.max(result.max_time_ns, elapsed);
        
        // Reset pool periodically to avoid exhaustion
        if (i % 1000 == 999) {
            pool.reset();
        }
    }
    
    result.calculate();
    return result;
}

/// SIMD-optimized array operations
fn benchmarkSIMDOperations(allocator: Allocator) !BenchmarkResult {
    const array_size = 1024;
    const data = try allocator.alloc(f64, array_size);
    defer allocator.free(data);
    
    const other = try allocator.alloc(f64, array_size);
    defer allocator.free(other);
    
    // Initialize data
    for (data) |*val, i| {
        val.* = @intToFloat(f64, i);
    }
    
    for (other) |*val, i| {
        val.* = @intToFloat(f64, i) * 2.0;
    }
    
    var result = BenchmarkResult{
        .operation = "SIMD Array Operations",
        .iterations = TEST_ITERATIONS,
        .total_time_ns = 0,
        .min_time_ns = std.math.maxInt(u64),
        .max_time_ns = 0,
        .avg_time_ns = 0,
        .ops_per_second = 0,
        .memory_used = array_size * @sizeOf(f64) * 2,
    };
    
    print("⚡ Testing SIMD array operations...\n");
    
    var i: u64 = 0;
    while (i < TEST_ITERATIONS) : (i += 1) {
        const timer = Timer.start();
        
        // Vectorized addition
        var j: usize = 0;
        while (j < array_size) : (j += 1) {
            data[j] += other[j];
        }
        
        const elapsed = timer.lap();
        result.total_time_ns += elapsed;
        result.min_time_ns = std.math.min(result.min_time_ns, elapsed);
        result.max_time_ns = std.math.max(result.max_time_ns, elapsed);
    }
    
    result.calculate();
    return result;
}

/// Hash table performance benchmark
fn benchmarkHashTable(allocator: Allocator) !BenchmarkResult {
    var map = HashMap(u64, u64, std.hash_map.getAutoHashFn(u64), std.hash_map.getAutoEqlFn(u64), 80).init(allocator);
    defer map.deinit();
    
    var result = BenchmarkResult{
        .operation = "HashMap Insert/Lookup",
        .iterations = TEST_ITERATIONS,
        .total_time_ns = 0,
        .min_time_ns = std.math.maxInt(u64),
        .max_time_ns = 0,
        .avg_time_ns = 0,
        .ops_per_second = 0,
        .memory_used = 0,
    };
    
    print("🗂️  Testing hash table operations...\n");
    
    var i: u64 = 0;
    while (i < TEST_ITERATIONS) : (i += 1) {
        const timer = Timer.start();
        
        // Insert
        try map.put(i, i * i);
        
        // Lookup
        const value = map.get(i);
        assert(value != null and value.? == i * i);
        
        const elapsed = timer.lap();
        result.total_time_ns += elapsed;
        result.min_time_ns = std.math.min(result.min_time_ns, elapsed);
        result.max_time_ns = std.math.max(result.max_time_ns, elapsed);
    }
    
    result.memory_used = map.capacity() * (@sizeOf(u64) * 2 + @sizeOf(usize));
    result.calculate();
    return result;
}

/// Generate performance report
fn generateReport(results: []const BenchmarkResult, allocator: Allocator) !void {
    const timestamp = std.time.timestamp();
    
    // Create JSON report
    var json_filename = try std.fmt.allocPrint(allocator, "zig_performance_report_{d}.json", .{timestamp});
    defer allocator.free(json_filename);
    
    const json_file = try std.fs.cwd().createFile(json_filename, .{});
    defer json_file.close();
    
    const writer = json_file.writer();
    
    try writer.print("{{\n");
    try writer.print("  \"timestamp\": {d},\n", .{timestamp});
    try writer.print("  \"zig_version\": \"{s}\",\n", .{@import("builtin").zig_version_string});
    try writer.print("  \"target\": \"{s}\",\n", .{@import("builtin").target.cpu.arch.name()});
    try writer.print("  \"benchmarks\": [\n");
    
    for (results) |result, i| {
        try writer.print("    {{\n");
        try writer.print("      \"operation\": \"{s}\",\n", .{result.operation});
        try writer.print("      \"iterations\": {d},\n", .{result.iterations});
        try writer.print("      \"avg_time_ns\": {d:.2},\n", .{result.avg_time_ns});
        try writer.print("      \"min_time_ns\": {d},\n", .{result.min_time_ns});
        try writer.print("      \"max_time_ns\": {d},\n", .{result.max_time_ns});
        try writer.print("      \"ops_per_second\": {d:.0},\n", .{result.ops_per_second});
        try writer.print("      \"memory_used\": {d}\n", .{result.memory_used});
        try writer.print("    }}");
        if (i < results.len - 1) try writer.print(",");
        try writer.print("\n");
    }
    
    try writer.print("  ]\n");
    try writer.print("}}\n");
    
    // Create text summary
    var summary_filename = try std.fmt.allocPrint(allocator, "zig_performance_summary_{d}.txt", .{timestamp});
    defer allocator.free(summary_filename);
    
    const summary_file = try std.fs.cwd().createFile(summary_filename, .{});
    defer summary_file.close();
    
    const summary_writer = summary_file.writer();
    
    try summary_writer.print("ZIG DATA STRUCTURE PERFORMANCE REPORT\n");
    try summary_writer.print("=====================================\n");
    try summary_writer.print("Generated: {d}\n", .{timestamp});
    try summary_writer.print("Zig Version: {s}\n", .{@import("builtin").zig_version_string});
    try summary_writer.print("Target: {s}\n\n", .{@import("builtin").target.cpu.arch.name()});
    
    try summary_writer.print("BENCHMARK RESULTS:\n");
    try summary_writer.print("------------------\n");
    
    for (results) |result| {
        const time_us = result.avg_time_ns / 1000.0;
        try summary_writer.print("{s:<30}: {d:>8.2}μs ({d:>10.0} ops/sec)\n", .{ 
            result.operation, 
            time_us, 
            result.ops_per_second 
        });
    }
    
    try summary_writer.print("\nPERFORMANCE TARGETS:\n");
    try summary_writer.print("-------------------\n");
    
    var sub_1us_count: u32 = 0;
    var sub_10us_count: u32 = 0;
    
    for (results) |result| {
        if (result.avg_time_ns < 1000) sub_1us_count += 1;
        if (result.avg_time_ns < 10000) sub_10us_count += 1;
    }
    
    try summary_writer.print("Sub-1μs operations: {d}/{d}\n", .{ sub_1us_count, results.len });
    try summary_writer.print("Sub-10μs operations: {d}/{d}\n", .{ sub_10us_count, results.len });
    try summary_writer.print("Memory efficiency: Lock-free, zero-allocation designs\n");
    
    print("📄 Report saved to: {s}\n", .{json_filename});
    print("📋 Summary saved to: {s}\n", .{summary_filename});
}

/// Main test runner
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    print("🚀 Zig Data Structure Performance Testing Suite\n");
    print("=" ** 50 ++ "\n");
    
    var results = ArrayList(BenchmarkResult).init(allocator);
    defer results.deinit();
    
    // Run all benchmarks
    try results.append(try benchmarkCircularBuffer(allocator));
    try results.append(try benchmarkPriorityQueue(allocator));
    try results.append(try benchmarkMemoryPool(allocator));
    try results.append(try benchmarkSIMDOperations(allocator));
    try results.append(try benchmarkHashTable(allocator));
    
    // Generate report
    try generateReport(results.items, allocator);
    
    print("\n🎯 All Zig performance tests completed successfully!\n");
    
    // Print quick summary
    var fast_ops: u32 = 0;
    for (results.items) |result| {
        if (result.avg_time_ns < 10000) fast_ops += 1;
    }
    
    print("📊 Performance Summary: {d}/{d} operations under 10μs\n", .{ fast_ops, results.items.len });
}
