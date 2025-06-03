// Simplified crypto utils without external dependencies
use std::collections::HashMap;

#[no_mangle]
pub extern "C" fn simple_hash(data: *const u8, len: usize) -> u64 {
    let slice = unsafe { std::slice::from_raw_parts(data, len) };
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in slice {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[no_mangle]
pub extern "C" fn verify_signature_simple(msg: *const u8, msg_len: usize) -> bool {
    // Placeholder implementation
    unsafe { std::slice::from_raw_parts(msg, msg_len).len() > 0 }
}

#[no_mangle]
pub extern "C" fn crypto_benchmark() -> u64 {
    let start = std::time::Instant::now();
    for i in 0..10000 {
        let data = i.to_be_bytes();
        simple_hash(data.as_ptr(), data.len());
    }
    start.elapsed().as_nanos() as u64
}
