//! A Rust implementation with the SuperInstance polyformalism canary wired in.
//!
//! The crate previously shipped as a binary with no `lib.rs` while its Cargo.toml and
//! README described it as a library. This file makes the description true and gives the
//! crate something for a canary test to stand on.

/// FNV-1a 64 - the digest every substrate in the SuperInstance fleet agrees on.
pub const FNV_OFFSET: u64 = 0xcbf29ce484222325;
pub const FNV_PRIME: u64 = 0x100000001b3;

#[inline]
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    for &b in bytes { h = (h ^ b as u64).wrapping_mul(FNV_PRIME); }
    h
}

/// The fleet canary, 0x024a555471370b18d.
pub const CANARY: u64 = 0x024a555471370b18d;

/// True if this crate's FNV-1a still agrees with the rest of the fleet.
pub fn canary_holds() -> bool { fnv1a64("café Δ 日本語".as_bytes()) == CANARY }
