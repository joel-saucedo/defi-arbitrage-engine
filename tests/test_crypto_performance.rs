/*
 * ═══════════════════════════════════════════════════════════════════════════════════
 *                         ETHEREUM MEV RESEARCH - CRYPTO TEST SUITE
 *                              Cryptographic Operations Testing
 *                                       Rust Implementation
 * ═══════════════════════════════════════════════════════════════════════════════════
 */

use std::time::{Duration, Instant};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// Import our crypto utilities (assuming they're in a lib)
// In actual implementation, this would import from our crypto_utils.rs
mod crypto_utils;
use crypto_utils::*;

#[cfg(test)]
mod tests {
    use super::*;

    const PERFORMANCE_ITERATIONS: usize = 10_000;
    const BATCH_SIZE: usize = 1_000;

    struct PerformanceMetrics {
        operation: String,
        iterations: usize,
        total_duration: Duration,
        mean_latency: Duration,
        min_latency: Duration,
        max_latency: Duration,
        p95_latency: Duration,
        p99_latency: Duration,
        throughput: f64,
    }

    impl PerformanceMetrics {
        fn new(operation: String, durations: Vec<Duration>) -> Self {
            let total_duration: Duration = durations.iter().sum();
            let iterations = durations.len();
            
            let mut sorted_durations = durations.clone();
            sorted_durations.sort();
            
            let mean_latency = total_duration / iterations as u32;
            let min_latency = sorted_durations[0];
            let max_latency = sorted_durations[iterations - 1];
            let p95_latency = sorted_durations[(iterations as f64 * 0.95) as usize];
            let p99_latency = sorted_durations[(iterations as f64 * 0.99) as usize];
            let throughput = iterations as f64 / total_duration.as_secs_f64();

            Self {
                operation,
                iterations,
                total_duration,
                mean_latency,
                min_latency,
                max_latency,
                p95_latency,
                p99_latency,
                throughput,
            }
        }

        fn print_report(&self) {
            println!("\n🔐 Cryptographic Performance Report: {}", self.operation);
            println!("   Iterations:    {:,}", self.iterations);
            println!("   Mean latency:  {:?}", self.mean_latency);
            println!("   Min latency:   {:?}", self.min_latency);
            println!("   Max latency:   {:?}", self.max_latency);
            println!("   P95 latency:   {:?}", self.p95_latency);
            println!("   P99 latency:   {:?}", self.p99_latency);
            println!("   Throughput:    {:.0} ops/sec", self.throughput);
        }
    }

    #[test]
    fn test_keccak256_performance() {
        println!("\n🚀 Testing Keccak256 Performance");
        
        let engine = CryptoEngine::new();
        let test_data = b"ethereum_mev_research_high_performance_testing";
        let mut durations = Vec::with_capacity(PERFORMANCE_ITERATIONS);

        // Warmup
        for _ in 0..1000 {
            engine.keccak256_cached(test_data);
        }

        // Benchmark
        for i in 0..PERFORMANCE_ITERATIONS {
            let mut input = test_data.to_vec();
            input.extend_from_slice(&i.to_be_bytes()); // Make each input unique
            
            let start = Instant::now();
            let _hash = engine.keccak256_cached(&input);
            let duration = start.elapsed();
            durations.push(duration);
        }

        let metrics = PerformanceMetrics::new("Keccak256".to_string(), durations);
        metrics.print_report();

        // Assertions for sub-millisecond performance
        assert!(
            metrics.mean_latency.as_micros() < 100,
            "Keccak256 mean latency {} μs exceeds 100 μs target",
            metrics.mean_latency.as_micros()
        );
        assert!(
            metrics.p99_latency.as_micros() < 500,
            "Keccak256 P99 latency {} μs exceeds 500 μs target", 
            metrics.p99_latency.as_micros()
        );
        assert!(
            metrics.throughput > 50_000.0,
            "Keccak256 throughput {:.0} ops/sec below 50,000 target",
            metrics.throughput
        );
    }

    #[test]
    fn test_batch_keccak256_performance() {
        println!("\n⚡ Testing Batch Keccak256 Performance");
        
        let engine = CryptoEngine::new();
        let inputs: Vec<Vec<u8>> = (0..BATCH_SIZE)
            .map(|i| format!("batch_input_{}", i).into_bytes())
            .collect();
        let input_refs: Vec<&[u8]> = inputs.iter().map(|v| v.as_slice()).collect();

        let mut durations = Vec::with_capacity(100);

        // Benchmark batch operations
        for _ in 0..100 {
            let start = Instant::now();
            let _hashes = engine.batch_keccak256(&input_refs);
            let duration = start.elapsed();
            durations.push(duration);
        }

        let metrics = PerformanceMetrics::new("Batch Keccak256".to_string(), durations);
        metrics.print_report();

        let per_item_latency = metrics.mean_latency.as_nanos() / BATCH_SIZE as u128;
        println!("   Per-item latency: {} ns", per_item_latency);

        // Batch should be significantly faster per item
        assert!(
            per_item_latency < 50_000, // 50 μs per item
            "Batch Keccak256 per-item latency {} ns too high",
            per_item_latency
        );
    }

    #[test]
    fn test_signature_verification_performance() {
        println!("\n🔏 Testing Signature Verification Performance");
        
        let engine = CryptoEngine::new();
        
        // Generate test keypair and signature
        let secp = secp256k1::Secp256k1::new();
        let (secret_key, public_key) = secp.generate_keypair(&mut rand::thread_rng());
        
        let message = b"test_transaction_for_mev_research";
        let message_hash = engine.keccak256_cached(message);
        
        let signature = secp.sign_ecdsa_recoverable(
            &secp256k1::Message::from_slice(&message_hash).unwrap(),
            &secret_key
        );
        let (rec_id, sig_bytes) = signature.serialize_compact();
        
        // Create 65-byte signature with recovery ID
        let mut full_signature = [0u8; 65];
        full_signature[..64].copy_from_slice(&sig_bytes);
        full_signature[64] = rec_id.to_i32() as u8 + 27;

        let mut durations = Vec::with_capacity(PERFORMANCE_ITERATIONS / 10);

        // Benchmark verification (fewer iterations as it's more expensive)
        for _ in 0..(PERFORMANCE_ITERATIONS / 10) {
            let start = Instant::now();
            let _valid = engine.verify_signature(message, &full_signature, &public_key);
            let duration = start.elapsed();
            durations.push(duration);
        }

        let metrics = PerformanceMetrics::new("Signature Verification".to_string(), durations);
        metrics.print_report();

        assert!(
            metrics.mean_latency.as_micros() < 1000, // 1ms target
            "Signature verification mean latency {} μs exceeds 1000 μs target",
            metrics.mean_latency.as_micros()
        );
        assert!(
            metrics.throughput > 1_000.0,
            "Signature verification throughput {:.0} ops/sec below 1,000 target",
            metrics.throughput
        );
    }

    #[test]
    fn test_batch_signature_verification() {
        println!("\n📦 Testing Batch Signature Verification");
        
        let engine = CryptoEngine::new();
        let secp = secp256k1::Secp256k1::new();
        
        // Generate test data
        let batch_size = 100;
        let mut messages = Vec::with_capacity(batch_size);
        let mut signatures = Vec::with_capacity(batch_size);
        let mut public_keys = Vec::with_capacity(batch_size);
        
        for i in 0..batch_size {
            let (secret_key, public_key) = secp.generate_keypair(&mut rand::thread_rng());
            let message = format!("batch_message_{}", i).into_bytes();
            let message_hash = engine.keccak256_cached(&message);
            
            let signature = secp.sign_ecdsa_recoverable(
                &secp256k1::Message::from_slice(&message_hash).unwrap(),
                &secret_key
            );
            let (rec_id, sig_bytes) = signature.serialize_compact();
            
            let mut full_signature = [0u8; 65];
            full_signature[..64].copy_from_slice(&sig_bytes);
            full_signature[64] = rec_id.to_i32() as u8 + 27;
            
            messages.push(message);
            signatures.push(full_signature);
            public_keys.push(public_key);
        }
        
        let message_refs: Vec<&[u8]> = messages.iter().map(|m| m.as_slice()).collect();
        let signature_refs: Vec<&[u8]> = signatures.iter().map(|s| s.as_slice()).collect();
        let pubkey_refs: Vec<&secp256k1::PublicKey> = public_keys.iter().collect();

        let mut durations = Vec::with_capacity(100);

        // Benchmark batch verification
        for _ in 0..100 {
            let start = Instant::now();
            let results = engine.batch_verify_signatures(&message_refs, &signature_refs, &pubkey_refs);
            let duration = start.elapsed();
            durations.push(duration);
            
            // Verify all signatures passed
            assert_eq!(results.len(), batch_size);
            assert!(results.iter().all(|&valid| valid), "Some signatures failed verification");
        }

        let metrics = PerformanceMetrics::new("Batch Signature Verification".to_string(), durations);
        metrics.print_report();

        let per_signature_latency = metrics.mean_latency.as_micros() / batch_size as u128;
        println!("   Per-signature latency: {} μs", per_signature_latency);

        assert!(
            per_signature_latency < 500, // 500 μs per signature in batch
            "Batch signature verification per-item latency {} μs too high",
            per_signature_latency
        );
    }

    #[test]
    fn test_merkle_tree_performance() {
        println!("\n🌲 Testing Merkle Tree Performance");
        
        // Generate test leaves
        let leaf_count = 1024;
        let leaves: Vec<[u8; 32]> = (0..leaf_count)
            .map(|i| {
                let mut leaf = [0u8; 32];
                leaf[28..32].copy_from_slice(&(i as u32).to_be_bytes());
                leaf
            })
            .collect();

        let mut construction_durations = Vec::with_capacity(100);
        let mut proof_durations = Vec::with_capacity(1000);

        // Benchmark tree construction
        for _ in 0..100 {
            let start = Instant::now();
            let _tree = MerkleTree::new(&leaves);
            let duration = start.elapsed();
            construction_durations.push(duration);
        }

        let construction_metrics = PerformanceMetrics::new(
            "Merkle Tree Construction".to_string(), 
            construction_durations
        );
        construction_metrics.print_report();

        // Benchmark proof generation
        let tree = MerkleTree::new(&leaves);
        for _ in 0..1000 {
            let leaf_index = rand::random::<usize>() % leaf_count;
            let start = Instant::now();
            let _proof = tree.generate_proof(leaf_index);
            let duration = start.elapsed();
            proof_durations.push(duration);
        }

        let proof_metrics = PerformanceMetrics::new(
            "Merkle Proof Generation".to_string(),
            proof_durations
        );
        proof_metrics.print_report();

        assert!(
            construction_metrics.mean_latency.as_millis() < 10,
            "Merkle tree construction too slow: {} ms",
            construction_metrics.mean_latency.as_millis()
        );
        assert!(
            proof_metrics.mean_latency.as_micros() < 100,
            "Merkle proof generation too slow: {} μs",
            proof_metrics.mean_latency.as_micros()
        );
    }

    #[test]
    fn test_cache_effectiveness() {
        println!("\n💾 Testing Cache Effectiveness");
        
        let engine = CryptoEngine::new();
        let test_inputs = vec![
            b"repeated_input_1".as_slice(),
            b"repeated_input_2".as_slice(),
            b"repeated_input_3".as_slice(),
        ];

        // First pass - populate cache
        let start_cold = Instant::now();
        for input in &test_inputs {
            for _ in 0..1000 {
                engine.keccak256_cached(input);
            }
        }
        let cold_duration = start_cold.elapsed();

        // Second pass - use cache
        let start_warm = Instant::now();
        for input in &test_inputs {
            for _ in 0..1000 {
                engine.keccak256_cached(input);
            }
        }
        let warm_duration = start_warm.elapsed();

        let speedup = cold_duration.as_nanos() as f64 / warm_duration.as_nanos() as f64;

        println!("   Cold cache duration: {:?}", cold_duration);
        println!("   Warm cache duration: {:?}", warm_duration);
        println!("   Cache speedup:       {:.2}x", speedup);

        let (addr_cache_size, hash_cache_size) = engine.cache_stats();
        println!("   Address cache size:  {}", addr_cache_size);
        println!("   Hash cache size:     {}", hash_cache_size);

        assert!(
            speedup > 2.0,
            "Cache speedup {:.2}x insufficient (target: > 2.0x)",
            speedup
        );
        assert!(
            hash_cache_size >= test_inputs.len(),
            "Hash cache not properly populated"
        );
    }

    #[test]
    fn test_concurrent_crypto_operations() {
        println!("\n🔄 Testing Concurrent Crypto Operations");
        
        let engine = Arc::new(CryptoEngine::new());
        let results = Arc::new(Mutex::new(Vec::new()));
        let thread_count = 8;
        let operations_per_thread = 1000;

        let start = Instant::now();
        
        std::thread::scope(|s| {
            for thread_id in 0..thread_count {
                let engine_ref = Arc::clone(&engine);
                let results_ref = Arc::clone(&results);
                
                s.spawn(move || {
                    let mut thread_durations = Vec::with_capacity(operations_per_thread);
                    
                    for i in 0..operations_per_thread {
                        let input = format!("thread_{}_operation_{}", thread_id, i);
                        let start = Instant::now();
                        let _hash = engine_ref.keccak256_cached(input.as_bytes());
                        let duration = start.elapsed();
                        thread_durations.push(duration);
                    }
                    
                    results_ref.lock().unwrap().extend(thread_durations);
                });
            }
        });

        let total_duration = start.elapsed();
        let all_durations = results.lock().unwrap().clone();
        
        let total_operations = thread_count * operations_per_thread;
        let throughput = total_operations as f64 / total_duration.as_secs_f64();
        
        println!("   Threads:           {}", thread_count);
        println!("   Operations/thread: {}", operations_per_thread);
        println!("   Total operations:  {}", total_operations);
        println!("   Total duration:    {:?}", total_duration);
        println!("   Throughput:        {:.0} ops/sec", throughput);
        
        let concurrent_metrics = PerformanceMetrics::new(
            "Concurrent Crypto Operations".to_string(),
            all_durations
        );
        concurrent_metrics.print_report();

        assert!(
            throughput > 100_000.0,
            "Concurrent throughput {:.0} ops/sec below 100,000 target",
            throughput
        );
    }

    #[test]
    fn test_memory_usage_optimization() {
        println!("\n🧠 Testing Memory Usage Optimization");
        
        let engine = CryptoEngine::new();
        
        // Perform many operations to stress memory usage
        for i in 0..10_000 {
            let input = format!("memory_test_{}", i);
            engine.keccak256_cached(input.as_bytes());
        }
        
        let (addr_cache_size, hash_cache_size) = engine.cache_stats();
        
        println!("   Operations performed: 10,000");
        println!("   Address cache size:   {}", addr_cache_size);
        println!("   Hash cache size:      {}", hash_cache_size);
        
        // Test cache clearing
        engine.clear_caches();
        let (addr_cache_after, hash_cache_after) = engine.cache_stats();
        
        println!("   After cache clear:");
        println!("   Address cache size:   {}", addr_cache_after);
        println!("   Hash cache size:      {}", hash_cache_after);
        
        assert_eq!(addr_cache_after, 0, "Address cache not properly cleared");
        assert_eq!(hash_cache_after, 0, "Hash cache not properly cleared");
        
        // Ensure cache doesn't grow unbounded
        assert!(
            hash_cache_size <= 10_000,
            "Hash cache size {} suggests unbounded growth",
            hash_cache_size
        );
    }

    #[test]
    fn test_nonce_generation_performance() {
        println!("\n🎲 Testing Nonce Generation Performance");
        
        let engine = CryptoEngine::new();
        let seed = b"mev_research_nonce_seed";
        let mut durations = Vec::with_capacity(PERFORMANCE_ITERATIONS);

        // Benchmark nonce generation
        for counter in 0..PERFORMANCE_ITERATIONS {
            let start = Instant::now();
            let _nonce = engine.generate_nonce(seed, counter as u64);
            let duration = start.elapsed();
            durations.push(duration);
        }

        let metrics = PerformanceMetrics::new("Nonce Generation".to_string(), durations);
        metrics.print_report();

        assert!(
            metrics.mean_latency.as_micros() < 50,
            "Nonce generation mean latency {} μs exceeds 50 μs target",
            metrics.mean_latency.as_micros()
        );
        assert!(
            metrics.throughput > 100_000.0,
            "Nonce generation throughput {:.0} ops/sec below 100,000 target",
            metrics.throughput
        );
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_full_transaction_crypto_pipeline() {
        println!("\n🔗 Testing Full Transaction Crypto Pipeline");
        
        let engine = CryptoEngine::new();
        let secp = secp256k1::Secp256k1::new();
        
        // Simulate a complete MEV transaction processing pipeline
        let transaction_count = 1000;
        let start = Instant::now();
        
        for i in 0..transaction_count {
            // 1. Generate transaction data
            let tx_data = format!("{{\"to\":\"0x123\",\"value\":{},\"nonce\":{}}}", i * 1000, i);
            
            // 2. Hash transaction
            let tx_hash = engine.keccak256_cached(tx_data.as_bytes());
            
            // 3. Generate nonce for MEV opportunity
            let mev_nonce = engine.generate_nonce(&tx_hash, i as u64);
            
            // 4. Create signature (simplified)
            let (secret_key, public_key) = secp.generate_keypair(&mut rand::thread_rng());
            let signature = secp.sign_ecdsa_recoverable(
                &secp256k1::Message::from_slice(&tx_hash).unwrap(),
                &secret_key
            );
            
            // 5. Extract address from public key
            let _address = engine.pubkey_to_address(&public_key);
            
            // Verify we can process at high speed
            if i % 100 == 0 {
                let elapsed = start.elapsed();
                let rate = (i + 1) as f64 / elapsed.as_secs_f64();
                if rate < 1000.0 {
                    panic!("Transaction processing rate {} tx/sec too slow", rate);
                }
            }
        }
        
        let total_duration = start.elapsed();
        let throughput = transaction_count as f64 / total_duration.as_secs_f64();
        
        println!("   Transactions processed: {}", transaction_count);
        println!("   Total duration:         {:?}", total_duration);
        println!("   Throughput:             {:.0} tx/sec", throughput);
        
        assert!(
            throughput > 5_000.0,
            "Pipeline throughput {:.0} tx/sec below 5,000 target",
            throughput
        );
    }
}
