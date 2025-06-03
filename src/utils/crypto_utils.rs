// ═══════════════════════════════════════════════════════════════════════════════════
//                         ETHEREUM MEV RESEARCH - CRYPTO UTILITIES
//                              High-Performance Cryptographic Operations
//                                    Rust Implementation
// ═══════════════════════════════════════════════════════════════════════════════════

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use sha3::{Digest, Keccak256};
use secp256k1::{PublicKey, SecretKey, Secp256k1};
use rayon::prelude::*;

/// Ultra-fast cryptographic utilities for MEV operations
/// Optimized for sub-millisecond execution with parallel processing
pub struct CryptoEngine {
    secp: Secp256k1<secp256k1::All>,
    address_cache: Arc<Mutex<HashMap<Vec<u8>, String>>>,
    hash_cache: Arc<Mutex<HashMap<Vec<u8>, [u8; 32]>>>,
}

impl CryptoEngine {
    pub fn new() -> Self {
        Self {
            secp: Secp256k1::new(),
            address_cache: Arc::new(Mutex::new(HashMap::with_capacity(10000))),
            hash_cache: Arc::new(Mutex::new(HashMap::with_capacity(50000))),
        }
    }

    /// High-speed Keccak256 hashing with caching for repeated operations
    pub fn keccak256_cached(&self, input: &[u8]) -> [u8; 32] {
        {
            let cache = self.hash_cache.lock().unwrap();
            if let Some(&hash) = cache.get(input) {
                return hash;
            }
        }

        let mut hasher = Keccak256::new();
        hasher.update(input);
        let result = hasher.finalize();
        let hash: [u8; 32] = result.into();

        let mut cache = self.hash_cache.lock().unwrap();
        cache.insert(input.to_vec(), hash);
        hash
    }

    /// Vectorized batch hashing for transaction processing
    pub fn batch_keccak256(&self, inputs: &[&[u8]]) -> Vec<[u8; 32]> {
        inputs.par_iter()
            .map(|&input| {
                let mut hasher = Keccak256::new();
                hasher.update(input);
                let result = hasher.finalize();
                result.into()
            })
            .collect()
    }

    /// Extract Ethereum address from public key with caching
    pub fn pubkey_to_address(&self, pubkey: &PublicKey) -> String {
        let serialized = pubkey.serialize_uncompressed();
        
        {
            let cache = self.address_cache.lock().unwrap();
            if let Some(addr) = cache.get(&serialized.to_vec()) {
                return addr.clone();
            }
        }

        // Skip first byte (0x04) and hash the 64-byte coordinate pair
        let hash = self.keccak256_cached(&serialized[1..]);
        
        // Take last 20 bytes for Ethereum address
        let address = format!("0x{}", hex::encode(&hash[12..]));
        
        let mut cache = self.address_cache.lock().unwrap();
        cache.insert(serialized.to_vec(), address.clone());
        address
    }

    /// High-performance signature verification for MEV transactions
    pub fn verify_signature(&self, message: &[u8], signature: &[u8], pubkey: &PublicKey) -> bool {
        if signature.len() != 65 {
            return false;
        }

        let message_hash = self.keccak256_cached(message);
        
        // Ethereum uses recoverable signatures
        let recovery_id = secp256k1::ecdsa::RecoveryId::from_i32(signature[64] as i32 - 27);
        if recovery_id.is_err() {
            return false;
        }

        let sig = secp256k1::ecdsa::RecoverableSignature::from_compact(
            &signature[..64], 
            recovery_id.unwrap()
        );
        
        if sig.is_err() {
            return false;
        }

        let recovered = self.secp.recover_ecdsa(&secp256k1::Message::from_slice(&message_hash).unwrap(), &sig.unwrap());
        
        match recovered {
            Ok(recovered_pubkey) => recovered_pubkey == *pubkey,
            Err(_) => false,
        }
    }

    /// Batch signature verification for high-throughput processing
    pub fn batch_verify_signatures(
        &self, 
        messages: &[&[u8]], 
        signatures: &[&[u8]], 
        pubkeys: &[&PublicKey]
    ) -> Vec<bool> {
        assert_eq!(messages.len(), signatures.len());
        assert_eq!(signatures.len(), pubkeys.len());

        (0..messages.len()).into_par_iter()
            .map(|i| self.verify_signature(messages[i], signatures[i], pubkeys[i]))
            .collect()
    }

    /// Generate deterministic nonce for MEV transaction ordering
    pub fn generate_nonce(&self, seed: &[u8], counter: u64) -> [u8; 32] {
        let mut input = Vec::with_capacity(seed.len() + 8);
        input.extend_from_slice(seed);
        input.extend_from_slice(&counter.to_be_bytes());
        self.keccak256_cached(&input)
    }

    /// Clear caches for memory management
    pub fn clear_caches(&self) {
        self.address_cache.lock().unwrap().clear();
        self.hash_cache.lock().unwrap().clear();
    }

    /// Get cache statistics for monitoring
    pub fn cache_stats(&self) -> (usize, usize) {
        let addr_count = self.address_cache.lock().unwrap().len();
        let hash_count = self.hash_cache.lock().unwrap().len();
        (addr_count, hash_count)
    }
}

/// Thread-safe singleton instance for global access
lazy_static::lazy_static! {
    pub static ref CRYPTO_ENGINE: CryptoEngine = CryptoEngine::new();
}

/// Fast Merkle tree implementation for transaction batching
pub struct MerkleTree {
    nodes: Vec<[u8; 32]>,
    leaf_count: usize,
}

impl MerkleTree {
    pub fn new(leaves: &[[u8; 32]]) -> Self {
        let leaf_count = leaves.len();
        let total_nodes = 2 * leaf_count.next_power_of_two() - 1;
        let mut nodes = vec![[0u8; 32]; total_nodes];
        
        // Copy leaves to the bottom level
        for (i, &leaf) in leaves.iter().enumerate() {
            nodes[total_nodes - leaf_count + i] = leaf;
        }
        
        // Build tree bottom-up with parallel processing
        let mut level_start = total_nodes - leaf_count.next_power_of_two();
        while level_start > 0 {
            let level_size = (total_nodes - level_start) / 2;
            (0..level_size).into_par_iter().for_each(|i| {
                let left_idx = level_start + 2 * i;
                let right_idx = left_idx + 1;
                let parent_idx = level_start / 2 + i;
                
                let mut hasher = Keccak256::new();
                hasher.update(&nodes[left_idx]);
                hasher.update(&nodes[right_idx]);
                nodes[parent_idx] = hasher.finalize().into();
            });
            level_start /= 2;
        }
        
        Self { nodes, leaf_count }
    }
    
    pub fn root(&self) -> [u8; 32] {
        self.nodes[0]
    }
    
    pub fn generate_proof(&self, leaf_index: usize) -> Vec<[u8; 32]> {
        if leaf_index >= self.leaf_count {
            return vec![];
        }
        
        let mut proof = Vec::new();
        let mut current_idx = self.nodes.len() - self.leaf_count.next_power_of_two() + leaf_index;
        
        while current_idx > 0 {
            let sibling_idx = if current_idx % 2 == 0 { current_idx + 1 } else { current_idx - 1 };
            if sibling_idx < self.nodes.len() {
                proof.push(self.nodes[sibling_idx]);
            }
            current_idx = (current_idx - 1) / 2;
        }
        
        proof
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keccak256_consistency() {
        let engine = CryptoEngine::new();
        let input = b"test input";
        let hash1 = engine.keccak256_cached(input);
        let hash2 = engine.keccak256_cached(input);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_batch_hashing_performance() {
        let engine = CryptoEngine::new();
        let inputs: Vec<&[u8]> = (0..1000).map(|i| format!("input_{}", i).as_bytes()).collect();
        let hashes = engine.batch_keccak256(&inputs);
        assert_eq!(hashes.len(), 1000);
    }

    #[test]
    fn test_merkle_tree_construction() {
        let leaves: Vec<[u8; 32]> = (0..8).map(|i| {
            let mut leaf = [0u8; 32];
            leaf[31] = i;
            leaf
        }).collect();
        
        let tree = MerkleTree::new(&leaves);
        let root = tree.root();
        assert_ne!(root, [0u8; 32]);
        
        let proof = tree.generate_proof(0);
        assert!(!proof.is_empty());
    }
}
