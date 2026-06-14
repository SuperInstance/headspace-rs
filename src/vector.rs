//! ARM-optimised vector operations for headspace-rs.
//!
//! Uses NEON SIMD intrinsics (`core::arch::aarch64`) for AArch64 dot
//! product, cosine similarity, L2 distance, and nearest-neighbour search.
//! All SIMD paths are gated behind `#[cfg(target_feature = "neon")]`.
//!
//! A scalar fallback is provided for non-NEON targets.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// NEON intrinsic helpers
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn neon_dot_product_impl(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::aarch64::*;

    let n = a.len();
    let mut i = 0;

    // Accumulate in a 128-bit NEON register (2 x float64).
    // Using `vpadd_f32` for pair-wise reduction at the end.
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);

    // Process 8 floats (2x q-registers) per iteration.
    while i + 8 <= n {
        let va0 = vld1q_f32(a[i..].as_ptr());
        let va1 = vld1q_f32(a[i + 4..].as_ptr());
        let vb0 = vld1q_f32(b[i..].as_ptr());
        let vb1 = vld1q_f32(b[i + 4..].as_ptr());

        acc0 = vfmaq_f32(acc0, va0, vb0);
        acc1 = vfmaq_f32(acc1, va1, vb1);
        i += 8;
    }

    // Process remaining 4 floats.
    if i + 4 <= n {
        let va = vld1q_f32(a[i..].as_ptr());
        let vb = vld1q_f32(b[i..].as_ptr());
        acc0 = vfmaq_f32(acc0, va, vb);
        i += 4;
    }

    // Reduce: acc0 + acc1 into one register
    let combined = vaddq_f32(acc0, acc1);

    // Horizontal pair-wise add: [s0+s1, s2+s3, _, _]
    let reduced = vpaddq_f32(combined, combined);
    // One more level: [s0+s1+s2+s3, ...]
    let final_val = vpaddq_f32(reduced, reduced);

    // Extract lane 0
    let mut result: f32 = 0.0;
    core::ptr::copy_nonoverlapping(&final_val as *const _ as *const f32, &mut result, 1);

    // Scalar remainder (0-3 elements).
    for j in i..n {
        result += a[j] * b[j];
    }

    result
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn neon_dot_product_impl(a: &[f32], b: &[f32]) -> f32 {
    // Scalar fallback
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// OPTIMISED NEON DOT PRODUCT — the headline function.
///
/// This is the ARM-optimised entry point for dotting two f32 vectors.
/// On aarch64+neon it uses FMAC fused multiply-add via the 128-bit NEON
/// pipeline.  The CPU can dispatch two FMAC µops per cycle on Neoverse N1,
/// giving ~8 FLOP/cycle throughput.
///
/// Panics if slices differ in length.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "dot_product: slice lengths must match ({} vs {})",
        a.len(),
        b.len()
    );
    // Safety: NEON intrinsics are safe under aarch64+neon.
    unsafe { neon_dot_product_impl(a, b) }
}

// ---------------------------------------------------------------------------
// Cosine similarity
// ---------------------------------------------------------------------------

/// Cosine similarity between two f32 slices.
///
/// Returns values in [-1.0, 1.0] where 1.0 = identical direction,
/// 0.0 = orthogonal, -1.0 = opposed.  Returns 0.0 if either vector
/// is zero-magnitude.
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product(a, b);
    let norm_a_sq = dot_product(a, a);
    let norm_b_sq = dot_product(b, b);

    if norm_a_sq == 0.0 || norm_b_sq == 0.0 || dot == 0.0 {
        return 0.0;
    }

    (dot / (norm_a_sq * norm_b_sq).sqrt()).clamp(-1.0, 1.0)
}

// ---------------------------------------------------------------------------
// L2 (Euclidean) distance
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn neon_l2_squared_impl(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::aarch64::*;

    let n = a.len();
    let mut i = 0;
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);

    while i + 8 <= n {
        let va0 = vld1q_f32(a[i..].as_ptr());
        let va1 = vld1q_f32(a[i + 4..].as_ptr());
        let vb0 = vld1q_f32(b[i..].as_ptr());
        let vb1 = vld1q_f32(b[i + 4..].as_ptr());

        let d0 = vsubq_f32(va0, vb0);
        let d1 = vsubq_f32(va1, vb1);
        acc0 = vfmaq_f32(acc0, d0, d0);
        acc1 = vfmaq_f32(acc1, d1, d1);
        i += 8;
    }

    if i + 4 <= n {
        let va = vld1q_f32(a[i..].as_ptr());
        let vb = vld1q_f32(b[i..].as_ptr());
        let d = vsubq_f32(va, vb);
        acc0 = vfmaq_f32(acc0, d, d);
        i += 4;
    }

    let combined = vaddq_f32(acc0, acc1);
    let reduced = vpaddq_f32(combined, combined);
    let final_val = vpaddq_f32(reduced, reduced);
    let mut result: f32 = 0.0;
    core::ptr::copy_nonoverlapping(&final_val as *const _ as *const f32, &mut result, 1);

    for j in i..n {
        let d = a[j] - b[j];
        result += d * d;
    }
    result
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn neon_l2_squared_impl(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Squared L2 distance between two f32 slices (NEON-accelerated on aarch64).
#[allow(dead_code)]
#[inline]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    unsafe { neon_l2_squared_impl(a, b) }
}

/// L2 (Euclidean) distance.
#[allow(dead_code)]
#[inline]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    l2_squared(a, b).sqrt()
}

// ---------------------------------------------------------------------------
// Data types & nearest-neighbour search
// ---------------------------------------------------------------------------

/// A single stored segment (text + embedding vector + unique ID).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    pub id: String,
    pub text: String,
    pub embedding: Vec<f32>,
}

/// A query result pairing a segment with its similarity score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub text: String,
    pub score: f32,
}

/// Find top-k nearest neighbours to `query_embedding` using cosine similarity.
///
/// This performs a brute-force linear scan over all stored segments,
/// scoring each with the NEON-accelerated dot product.  For the prototype
/// scale (~10 000 segments, ~384 dims) this is well within real-time
/// constraints on a Neoverse N1 core.
pub fn nearest_neighbours(
    query_embedding: &[f32],
    segments: &[Segment],
    k: usize,
) -> Vec<SearchResult> {
    let k = k.min(segments.len());
    if k == 0 {
        return Vec::new();
    }

    // Build scored list: (similarity, index)
    let mut scored: Vec<(f32, usize)> = segments
        .iter()
        .enumerate()
        .map(|(idx, seg)| {
            let sim = cosine_similarity(query_embedding, &seg.embedding);
            // Use bitwise-reversed f32 for descending sort
            (sim, idx)
        })
        .collect();

    // Partial sort: top-k only, descending
    scored.select_nth_unstable_by(k - 1, |a, b| {
        b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
    });
    scored.truncate(k);

    // Sort the top-k for consistent output ordering
    scored.sort_unstable_by(|a, b| {
        b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
    });

    scored
        .into_iter()
        .map(|(score, idx)| SearchResult {
            id: segments[idx].id.clone(),
            text: segments[idx].text.clone(),
            score,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product_identity() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        assert!((dot_product(&a, &a) - 30.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((dot_product(&a, &b)).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_same() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_opposite() {
        let a = vec![1.0, 2.0];
        let b = vec![-1.0, -2.0];
        assert!((cosine_similarity(&a, &b) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_l2_zero() {
        let a = vec![3.0, 4.0];
        assert!((l2_distance(&a, &a)).abs() < 1e-6);
    }

    #[test]
    fn test_l2_3_4_5() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((l2_distance(&a, &b) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_nearest_neighbours() {
        let segments = vec![
            Segment {
                id: "a".into(),
                text: "hello world".into(),
                embedding: vec![1.0, 0.0, 0.0],
            },
            Segment {
                id: "b".into(),
                text: "goodbye world".into(),
                embedding: vec![0.0, 1.0, 0.0],
            },
            Segment {
                id: "c".into(),
                text: "something else".into(),
                embedding: vec![0.0, 0.0, 1.0],
            },
        ];

        let query: Vec<f32> = vec![0.9, 0.1, 0.0];
        let results = nearest_neighbours(&query, &segments, 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "a");
        assert_eq!(results[1].id, "b");
    }

    #[test]
    fn test_dot_product_large() {
        let a: Vec<f32> = (0..1000).map(|i| (i as f32).sin()).collect();
        let b: Vec<f32> = (0..1000).map(|i| (i as f32).cos()).collect();
        let result = dot_product(&a, &b);
        assert!(result.is_finite());
    }

    #[test]
    fn test_empty_query_no_panic() {
        let segments: Vec<Segment> = vec![];
        let query: Vec<f32> = vec![1.0, 0.0];
        let results = nearest_neighbours(&query, &segments, 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_l2_symmetry() {
        let a: Vec<f32> = (0..20).map(|i| i as f32 * 0.5).collect();
        let b: Vec<f32> = (0..20).map(|i| i as f32 * 0.5 + 1.0).collect();
        assert!((l2_distance(&a, &b) - l2_distance(&b, &a)).abs() < 1e-6);
    }

    /// Verify the NEON implementation matches a simple scalar reference.
    #[test]
    fn test_neon_matches_scalar_dot() {
        let a: Vec<f32> = (0..128).map(|i| (i as f32).sin()).collect();
        let b: Vec<f32> = (0..128).map(|i| (i as f32).cos()).collect();

        let neon_result = dot_product(&a, &b);
        let scalar_result: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

        assert!(
            (neon_result - scalar_result).abs() < 1e-4,
            "NEON dot deviates from scalar: {neon_result} vs {scalar_result}"
        );
    }

    /// Verify the NEON L2 matches scalar L2.
    #[test]
    fn test_neon_matches_scalar_l2() {
        let a: Vec<f32> = (0..64).map(|i| (i as f32).sin()).collect();
        let b: Vec<f32> = (0..64).map(|i| (i as f32).cos()).collect();

        let neon_dist = l2_distance(&a, &b);
        let scalar_dist: f32 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let d = x - y;
                d * d
            })
            .sum::<f32>()
            .sqrt();

        assert!(
            (neon_dist - scalar_dist).abs() < 1e-4,
            "NEON L2 deviates from scalar: {neon_dist} vs {scalar_dist}"
        );
    }
}
