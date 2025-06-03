/**
 * ═══════════════════════════════════════════════════════════════════════════════════
 *                         ETHEREUM MEV RESEARCH - DATA STRUCTURES
 *                              High-Performance Data Containers
 *                                     Zig Implementation
 * ═══════════════════════════════════════════════════════════════════════════════════
 */

const std = @import("std");
const print = std.debug.print;
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const HashMap = std.HashMap;
const Atomic = std.atomic.Atomic;

/// High-performance circular buffer for MEV transaction processing
pub fn CircularBuffer(comptime T: type) type {
    return struct {
        const Self = @This();
        
        buffer: []T,
        head: usize,
        tail: usize,
        capacity: usize,
        allocator: Allocator,
        
        pub fn init(allocator: Allocator, capacity: usize) !Self {
            const buffer = try allocator.alloc(T, capacity);
            return Self{
                .buffer = buffer,
                .head = 0,
                .tail = 0,
                .capacity = capacity,
                .allocator = allocator,
            };
        }
        
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.buffer);
        }
        
        pub fn push(self: *Self, item: T) bool {
            const next_tail = (self.tail + 1) % self.capacity;
            if (next_tail == self.head) {
                return false; // Buffer full
            }
            
            self.buffer[self.tail] = item;
            self.tail = next_tail;
            return true;
        }
        
        pub fn pop(self: *Self) ?T {
            if (self.head == self.tail) {
                return null; // Buffer empty
            }
            
            const item = self.buffer[self.head];
            self.head = (self.head + 1) % self.capacity;
            return item;
        }
        
        pub fn size(self: *const Self) usize {
            if (self.tail >= self.head) {
                return self.tail - self.head;
            } else {
                return self.capacity - self.head + self.tail;
            }
        }
        
        pub fn is_full(self: *const Self) bool {
            return ((self.tail + 1) % self.capacity) == self.head;
        }
        
        pub fn is_empty(self: *const Self) bool {
            return self.head == self.tail;
        }
        
        pub fn peek(self: *const Self) ?T {
            if (self.head == self.tail) {
                return null;
            }
            return self.buffer[self.head];
        }
        
        pub fn batch_push(self: *Self, items: []const T) usize {
            var pushed: usize = 0;
            for (items) |item| {
                if (self.push(item)) {
                    pushed += 1;
                } else {
                    break;
                }
            }
            return pushed;
        }
        
        pub fn batch_pop(self: *Self, items: []T) usize {
            var popped: usize = 0;
            for (items) |*item| {
                if (self.pop()) |value| {
                    item.* = value;
                    popped += 1;
                } else {
                    break;
                }
            }
            return popped;
        }
    };
}

/// Lock-free ring buffer for high-frequency trading
pub fn LockFreeRingBuffer(comptime T: type) type {
    return struct {
        const Self = @This();
        
        buffer: []T,
        head: Atomic(usize),
        tail: Atomic(usize),
        capacity: usize,
        allocator: Allocator,
        
        pub fn init(allocator: Allocator, capacity: usize) !Self {
            // Ensure capacity is power of 2 for efficient modulo
            const actual_capacity = std.math.ceilPowerOfTwo(usize, capacity) catch capacity;
            const buffer = try allocator.alloc(T, actual_capacity);
            
            return Self{
                .buffer = buffer,
                .head = Atomic(usize).init(0),
                .tail = Atomic(usize).init(0),
                .capacity = actual_capacity,
                .allocator = allocator,
            };
        }
        
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.buffer);
        }
        
        pub fn push(self: *Self, item: T) bool {
            const current_tail = self.tail.load(.Acquire);
            const next_tail = (current_tail + 1) & (self.capacity - 1);
            
            if (next_tail == self.head.load(.Acquire)) {
                return false; // Buffer full
            }
            
            self.buffer[current_tail] = item;
            self.tail.store(next_tail, .Release);
            return true;
        }
        
        pub fn pop(self: *Self) ?T {
            const current_head = self.head.load(.Acquire);
            if (current_head == self.tail.load(.Acquire)) {
                return null; // Buffer empty
            }
            
            const item = self.buffer[current_head];
            const next_head = (current_head + 1) & (self.capacity - 1);
            self.head.store(next_head, .Release);
            return item;
        }
        
        pub fn size(self: *const Self) usize {
            const tail = self.tail.load(.Acquire);
            const head = self.head.load(.Acquire);
            return (tail - head) & (self.capacity - 1);
        }
        
        pub fn is_empty(self: *const Self) bool {
            return self.head.load(.Acquire) == self.tail.load(.Acquire);
        }
        
        pub fn is_full(self: *const Self) bool {
            const tail = self.tail.load(.Acquire);
            const head = self.head.load(.Acquire);
            return ((tail + 1) & (self.capacity - 1)) == head;
        }
    };
}

/// High-performance priority queue for MEV opportunity ranking
pub fn PriorityQueue(comptime T: type, comptime compareFn: fn (T, T) bool) type {
    return struct {
        const Self = @This();
        
        items: ArrayList(T),
        
        pub fn init(allocator: Allocator) Self {
            return Self{
                .items = ArrayList(T).init(allocator),
            };
        }
        
        pub fn deinit(self: *Self) void {
            self.items.deinit();
        }
        
        pub fn insert(self: *Self, item: T) !void {
            try self.items.append(item);
            self.siftUp(self.items.items.len - 1);
        }
        
        pub fn extractMax(self: *Self) ?T {
            if (self.items.items.len == 0) return null;
            
            const max_item = self.items.items[0];
            const last_item = self.items.pop();
            
            if (self.items.items.len > 0) {
                self.items.items[0] = last_item;
                self.siftDown(0);
            }
            
            return max_item;
        }
        
        pub fn peek(self: *const Self) ?T {
            if (self.items.items.len == 0) return null;
            return self.items.items[0];
        }
        
        pub fn size(self: *const Self) usize {
            return self.items.items.len;
        }
        
        pub fn is_empty(self: *const Self) bool {
            return self.items.items.len == 0;
        }
        
        fn siftUp(self: *Self, start_index: usize) void {
            var index = start_index;
            while (index > 0) {
                const parent_index = (index - 1) / 2;
                if (!compareFn(self.items.items[index], self.items.items[parent_index])) {
                    break;
                }
                
                const temp = self.items.items[index];
                self.items.items[index] = self.items.items[parent_index];
                self.items.items[parent_index] = temp;
                
                index = parent_index;
            }
        }
        
        fn siftDown(self: *Self, start_index: usize) void {
            var index = start_index;
            const len = self.items.items.len;
            
            while (true) {
                var largest = index;
                const left_child = 2 * index + 1;
                const right_child = 2 * index + 2;
                
                if (left_child < len and compareFn(self.items.items[left_child], self.items.items[largest])) {
                    largest = left_child;
                }
                
                if (right_child < len and compareFn(self.items.items[right_child], self.items.items[largest])) {
                    largest = right_child;
                }
                
                if (largest == index) {
                    break;
                }
                
                const temp = self.items.items[index];
                self.items.items[index] = self.items.items[largest];
                self.items.items[largest] = temp;
                
                index = largest;
            }
        }
    };
}

/// Cache-optimized hash table for transaction tracking
pub fn FastHashMap(comptime K: type, comptime V: type) type {
    return struct {
        const Self = @This();
        const Entry = struct {
            key: K,
            value: V,
            hash: u64,
            occupied: bool,
        };
        
        entries: []Entry,
        capacity: usize,
        size: usize,
        allocator: Allocator,
        
        pub fn init(allocator: Allocator, initial_capacity: usize) !Self {
            const capacity = std.math.ceilPowerOfTwo(usize, initial_capacity) catch initial_capacity;
            const entries = try allocator.alloc(Entry, capacity);
            
            for (entries) |*entry| {
                entry.occupied = false;
            }
            
            return Self{
                .entries = entries,
                .capacity = capacity,
                .size = 0,
                .allocator = allocator,
            };
        }
        
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.entries);
        }
        
        fn hash(key: K) u64 {
            // Simple hash function - replace with better hash for production
            var hasher = std.hash.Wyhash.init(0);
            std.hash.autoHash(&hasher, key);
            return hasher.final();
        }
        
        pub fn put(self: *Self, key: K, value: V) !void {
            if (self.size >= self.capacity * 3 / 4) {
                try self.resize();
            }
            
            const key_hash = hash(key);
            var index = key_hash & (self.capacity - 1);
            
            while (self.entries[index].occupied) {
                if (std.meta.eql(self.entries[index].key, key)) {
                    self.entries[index].value = value;
                    return;
                }
                index = (index + 1) & (self.capacity - 1);
            }
            
            self.entries[index] = Entry{
                .key = key,
                .value = value,
                .hash = key_hash,
                .occupied = true,
            };
            self.size += 1;
        }
        
        pub fn get(self: *const Self, key: K) ?V {
            const key_hash = hash(key);
            var index = key_hash & (self.capacity - 1);
            
            while (self.entries[index].occupied) {
                if (std.meta.eql(self.entries[index].key, key)) {
                    return self.entries[index].value;
                }
                index = (index + 1) & (self.capacity - 1);
            }
            
            return null;
        }
        
        pub fn remove(self: *Self, key: K) bool {
            const key_hash = hash(key);
            var index = key_hash & (self.capacity - 1);
            
            while (self.entries[index].occupied) {
                if (std.meta.eql(self.entries[index].key, key)) {
                    self.entries[index].occupied = false;
                    self.size -= 1;
                    
                    // Rehash following entries
                    var next_index = (index + 1) & (self.capacity - 1);
                    while (self.entries[next_index].occupied) {
                        const entry = self.entries[next_index];
                        self.entries[next_index].occupied = false;
                        self.size -= 1;
                        
                        self.put(entry.key, entry.value) catch unreachable;
                        next_index = (next_index + 1) & (self.capacity - 1);
                    }
                    
                    return true;
                }
                index = (index + 1) & (self.capacity - 1);
            }
            
            return false;
        }
        
        fn resize(self: *Self) !void {
            const old_entries = self.entries;
            const old_capacity = self.capacity;
            
            self.capacity *= 2;
            self.entries = try self.allocator.alloc(Entry, self.capacity);
            self.size = 0;
            
            for (self.entries) |*entry| {
                entry.occupied = false;
            }
            
            for (old_entries) |entry| {
                if (entry.occupied) {
                    try self.put(entry.key, entry.value);
                }
            }
            
            self.allocator.free(old_entries);
        }
        
        pub fn get_size(self: *const Self) usize {
            return self.size;
        }
        
        pub fn get_capacity(self: *const Self) usize {
            return self.capacity;
        }
        
        pub fn load_factor(self: *const Self) f64 {
            return @intToFloat(f64, self.size) / @intToFloat(f64, self.capacity);
        }
    };
}

/// Memory-efficient bit vector for transaction flags
pub const BitVector = struct {
    const Self = @This();
    
    bits: []u64,
    bit_count: usize,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, bit_count: usize) !Self {
        const word_count = (bit_count + 63) / 64;
        const bits = try allocator.alloc(u64, word_count);
        std.mem.set(u64, bits, 0);
        
        return Self{
            .bits = bits,
            .bit_count = bit_count,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.allocator.free(self.bits);
    }
    
    pub fn set(self: *Self, index: usize) void {
        if (index >= self.bit_count) return;
        
        const word_index = index / 64;
        const bit_index = index % 64;
        self.bits[word_index] |= (@as(u64, 1) << @intCast(u6, bit_index));
    }
    
    pub fn clear(self: *Self, index: usize) void {
        if (index >= self.bit_count) return;
        
        const word_index = index / 64;
        const bit_index = index % 64;
        self.bits[word_index] &= ~(@as(u64, 1) << @intCast(u6, bit_index));
    }
    
    pub fn get(self: *const Self, index: usize) bool {
        if (index >= self.bit_count) return false;
        
        const word_index = index / 64;
        const bit_index = index % 64;
        return (self.bits[word_index] & (@as(u64, 1) << @intCast(u6, bit_index))) != 0;
    }
    
    pub fn toggle(self: *Self, index: usize) void {
        if (index >= self.bit_count) return;
        
        const word_index = index / 64;
        const bit_index = index % 64;
        self.bits[word_index] ^= (@as(u64, 1) << @intCast(u6, bit_index));
    }
    
    pub fn popcount(self: *const Self) usize {
        var count: usize = 0;
        for (self.bits) |word| {
            count += @popCount(u64, word);
        }
        return count;
    }
    
    pub fn clear_all(self: *Self) void {
        std.mem.set(u64, self.bits, 0);
    }
    
    pub fn set_all(self: *Self) void {
        std.mem.set(u64, self.bits, std.math.maxInt(u64));
        
        // Clear unused bits in the last word
        const last_word_bits = self.bit_count % 64;
        if (last_word_bits != 0) {
            const last_word_index = self.bits.len - 1;
            const mask = (@as(u64, 1) << @intCast(u6, last_word_bits)) - 1;
            self.bits[last_word_index] &= mask;
        }
    }
};

/// Transaction structure for MEV detection
pub const Transaction = struct {
    hash: [32]u8,
    from: [20]u8,
    to: [20]u8,
    value: u256,
    gas_price: u64,
    gas_limit: u64,
    nonce: u64,
    data: []u8,
    timestamp: u64,
    block_number: u64,
    
    pub fn compare_by_gas_price(a: Transaction, b: Transaction) bool {
        return a.gas_price > b.gas_price;
    }
    
    pub fn compare_by_value(a: Transaction, b: Transaction) bool {
        return a.value > b.value;
    }
};

/// MEV opportunity structure
pub const MEVOpportunity = struct {
    profit: u256,
    gas_cost: u64,
    transactions: []Transaction,
    strategy: Strategy,
    confidence: f64,
    timestamp: u64,
    
    pub const Strategy = enum {
        Arbitrage,
        Sandwich,
        Frontrun,
        Backrun,
        Liquidation,
    };
    
    pub fn net_profit(self: *const MEVOpportunity) i256 {
        return @intCast(i256, self.profit) - @intCast(i256, self.gas_cost);
    }
    
    pub fn compare_by_profit(a: MEVOpportunity, b: MEVOpportunity) bool {
        return a.net_profit() > b.net_profit();
    }
};

// Type aliases for common data structures
pub const TransactionBuffer = CircularBuffer(Transaction);
pub const TransactionQueue = LockFreeRingBuffer(Transaction);
pub const MEVQueue = PriorityQueue(MEVOpportunity, MEVOpportunity.compare_by_profit);
pub const TransactionCache = FastHashMap([32]u8, Transaction);

test "circular buffer operations" {
    const testing = std.testing;
    var buffer = try CircularBuffer(u32).init(testing.allocator, 4);
    defer buffer.deinit();
    
    try testing.expect(buffer.push(1));
    try testing.expect(buffer.push(2));
    try testing.expect(buffer.push(3));
    
    try testing.expectEqual(@as(?u32, 1), buffer.pop());
    try testing.expectEqual(@as(?u32, 2), buffer.pop());
    
    try testing.expect(buffer.push(4));
    try testing.expect(buffer.push(5));
    
    try testing.expectEqual(@as(usize, 2), buffer.size());
}

test "priority queue operations" {
    const testing = std.testing;
    
    const Item = struct {
        value: u32,
        
        fn compare(a: @This(), b: @This()) bool {
            return a.value > b.value;
        }
    };
    
    var queue = PriorityQueue(Item, Item.compare).init(testing.allocator);
    defer queue.deinit();
    
    try queue.insert(Item{ .value = 3 });
    try queue.insert(Item{ .value = 1 });
    try queue.insert(Item{ .value = 4 });
    try queue.insert(Item{ .value = 2 });
    
    try testing.expectEqual(@as(u32, 4), queue.extractMax().?.value);
    try testing.expectEqual(@as(u32, 3), queue.extractMax().?.value);
    try testing.expectEqual(@as(u32, 2), queue.extractMax().?.value);
    try testing.expectEqual(@as(u32, 1), queue.extractMax().?.value);
}

test "fast hash map operations" {
    const testing = std.testing;
    var map = try FastHashMap(u32, []const u8).init(testing.allocator, 16);
    defer map.deinit();
    
    try map.put(1, "one");
    try map.put(2, "two");
    try map.put(3, "three");
    
    try testing.expectEqualStrings("one", map.get(1).?);
    try testing.expectEqualStrings("two", map.get(2).?);
    try testing.expectEqual(@as(?[]const u8, null), map.get(4));
    
    try testing.expect(map.remove(2));
    try testing.expectEqual(@as(?[]const u8, null), map.get(2));
}
