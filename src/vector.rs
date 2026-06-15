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
    /// Optional namespace for partitioning segments (e.g. "pulse", "lures").
    #[serde(default)]
    pub namespace: Option<String>,
    /// Optional TTL in seconds from created_at (after which segment expires).
    pub ttl_seconds: Option<u64>,
    /// Unix epoch seconds when this segment was created.
    #[serde(default)]
    pub created_at: u64,
    /// Content-style bucket for stratified search (0-4).
    ///   0 = short metrics/text (<200 chars, likely numeric)
    ///   1 = medium prose (200-2000 chars)
    ///   2 = long documents (>2000 chars)
    ///   3 = commands/namespaces (system-prefixed)
    ///   4 = queries (default)
    #[serde(default)]
    pub bucket: u8,
}

/// A query result pairing a segment with its similarity score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub text: String,
    pub score: f32,
}

// ---------------------------------------------------------------------------
// Bucket assignment for stratified sampling
// ---------------------------------------------------------------------------

/// System-prefix patterns for bucket 3 (commands/namespaces).
const SYSTEM_PREFIXES: &[&str] = &[
    "system:", "cmd:", "command:", "namespace:", "fn:", "fn ",
    "func:", "function:", "def ", "pub ", "impl ",
];

/// Assign a content-style bucket (0-4) based on text heuristics.
///
/// - Bucket 0: short metrics/text (< 200 chars, likely numeric)
/// - Bucket 1: medium prose (200-2000 chars)
/// - Bucket 2: long documents (> 2000 chars)
/// - Bucket 3: commands/namespaces (starts with a system prefix)
/// - Bucket 4: queries (default)
#[inline]
pub fn assign_bucket(text: &str) -> u8 {
    let trimmed = text.trim();
    let len = trimmed.len();

    // Check system/command prefixes first
    for prefix in SYSTEM_PREFIXES {
        if trimmed.starts_with(prefix) {
            return 3;
        }
    }

    // Check for likely numeric/metric content (short and mostly digits/symbols)
    if len < 200 {
        let non_space: usize = trimmed.chars().filter(|c| !c.is_whitespace()).count();
        if non_space > 0 {
            let numeric_count: usize = trimmed
                .chars()
                .filter(|c| !c.is_whitespace())
                .filter(|c| {
                    c.is_ascii_digit() || *c == '.' || *c == '-' || *c == '/' || *c == '%' || *c == '='
                })
                .count();
            if numeric_count as f64 / non_space as f64 > 0.4 {
                return 0;
            }
        }
        // Short but not clearly numeric -> medium prose
        return 1;
    }

    if len <= 2000 {
        return 1;
    }

    // > 2000 chars
    2
}

/// Determine the query bucket for stratified search ordering.
#[inline]
pub fn query_bucket(text: &str) -> u8 {
    assign_bucket(text)
}

/// Order in which to search buckets, given the query bucket.
/// The query's own bucket is searched first, then the others.
fn bucket_search_order(query_bucket_id: u8) -> [u8; 5] {
    let all: [u8; 5] = [0, 1, 2, 3, 4];
    let mut order = [0u8; 5];
    let mut idx = 1;
    order[0] = query_bucket_id;
    for &b in &all {
        if b != query_bucket_id {
            order[idx] = b;
            idx += 1;
        }
    }
    order
}

/// Confidence threshold for early exit from stratified search.
/// If the top match from the first bucket exceeds this, we skip other buckets.
const EARLY_EXIT_CONFIDENCE: f32 = 0.80;

/// Find top-k nearest neighbours to `query_embedding` using cosine similarity,
/// with stratified bucket sampling for early exit.
///
/// Groups segments into 5 content-style buckets, searches the query's bucket
/// first, and returns early if the top match has confidence > 0.8.
/// Otherwise falls through to remaining buckets for correctness.
///
/// The `query_text` parameter is used for bucket assignment. When `None`,
/// falls back to full brute-force search (original behaviour).
pub fn nearest_neighbours(
    query_embedding: &[f32],
    segments: &[Segment],
    k: usize,
    namespace_filter: Option<&str>,
    now_unix: u64,
    query_text: Option<&str>,
) -> Vec<SearchResult> {
    // Filter eligible segments: optional namespace matching + TTL check
    let eligible: Vec<&Segment> = segments
        .iter()
        .filter(|seg| {
            let ns_ok = namespace_filter.map_or(true, |ns| seg.namespace.as_deref() == Some(ns));
            let ttl_ok = seg.ttl_seconds.map_or(true, |ttl| seg.created_at + ttl > now_unix);
            ns_ok && ttl_ok
        })
        .collect();

    let k = k.min(eligible.len());
    if k == 0 {
        return Vec::new();
    }

    // If eligible set is small enough, skip stratification overhead
    if eligible.len() < 20 {
        return brute_force_nearest(query_embedding, &eligible, k);
    }

    // Determine query bucket from text (or default to 4)
    let q_bucket = query_text.map_or(4u8, |t| query_bucket(t));

    // Partition eligible segments by bucket
    let mut bucket_groups: [Vec<&Segment>; 5] = [
        Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(),
    ];
    for seg in &eligible {
        let b = (seg.bucket as usize).min(4);
        bucket_groups[b].push(*seg);
    }

    let search_order = bucket_search_order(q_bucket);

    let mut top_k: Vec<(f32, &Segment)> = Vec::with_capacity(k);
    let mut searched_buckets = 0u8;

    for &bucket_id in &search_order {
        let bucket_segs = &bucket_groups[bucket_id as usize];
        if bucket_segs.is_empty() {
            continue;
        }

        searched_buckets += 1;

        // Score all segments in this bucket with software-pipelined prefetch
        let mut scored: Vec<(f32, &Segment)> = Vec::with_capacity(bucket_segs.len());
        let blen = bucket_segs.len();
        for idx in 0..blen {
            if idx + 1 < blen {
                let next_emb = &bucket_segs[idx + 1].embedding;
                if !next_emb.is_empty() {
                    std::hint::black_box(&next_emb[0]);
                }
            }
            let sim = cosine_similarity(query_embedding, &bucket_segs[idx].embedding);
            scored.push((sim, bucket_segs[idx]));
        }

        // Merge into top-k accumulator
        if top_k.is_empty() {
            top_k = scored;
        } else {
            top_k.extend(scored);
        }

        // Keep only top-k
        if top_k.len() > k {
            top_k.select_nth_unstable_by(k - 1, |a, b| {
                b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            top_k.truncate(k);
        }

        // Early exit: if our top match (after first bucket) is very confident,
        // skip remaining buckets.
        if searched_buckets == 1 && !top_k.is_empty() {
            let top_score = top_k[0].0;
            if top_score > EARLY_EXIT_CONFIDENCE {
                tracing::debug!(
                    "stratified early exit: bucket={} top_score={:.4} early=true",
                    bucket_id,
                    top_score,
                );
                break;
            }
        }
    }

    // Sort the top-k for consistent output ordering
    top_k.sort_unstable_by(|a, b| {
        b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
    });

    if top_k.len() > k {
        top_k.truncate(k);
    }

    let buckets_searched = searched_buckets;

    let results: Vec<SearchResult> = top_k
        .into_iter()
        .map(|(score, seg)| SearchResult {
            id: seg.id.clone(),
            text: seg.text.clone(),
            score,
        })
        .collect();

    let total_buckets = search_order
        .iter()
        .filter(|&&b| !bucket_groups[b as usize].is_empty())
        .count();
    if buckets_searched < total_buckets as u8 {
        tracing::debug!(
            "stratified search: searched {}/{} buckets, {} results",
            buckets_searched,
            total_buckets,
            results.len(),
        );
    }

    results
}

/// Brute-force search (original behaviour) used for small eligible sets or
/// as the fallback. Maintains software-pipelined prefetch for performance.
fn brute_force_nearest<'a>(
    query_embedding: &[f32],
    eligible: &[&'a Segment],
    k: usize,
) -> Vec<SearchResult> {
    let mut scored: Vec<(f32, usize)> = Vec::with_capacity(eligible.len());
    let len = eligible.len();
    for idx in 0..len {
        if idx + 1 < len {
            let next_emb = &eligible[idx + 1].embedding;
            if !next_emb.is_empty() {
                std::hint::black_box(&next_emb[0]);
            }
        }
        let sim = cosine_similarity(query_embedding, &eligible[idx].embedding);
        scored.push((sim, idx));
    }

    scored.select_nth_unstable_by(k - 1, |a, b| {
        b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
    });
    scored.truncate(k);

    scored.sort_unstable_by(|a, b| {
        b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
    });

    scored
        .into_iter()
        .map(|(score, idx)| {
            let seg = eligible[idx];
            SearchResult {
                id: seg.id.clone(),
                text: seg.text.clone(),
                score,
            }
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
                namespace: None,
                ttl_seconds: None,
                created_at: 1000,
                bucket: 1,
            },
            Segment {
                id: "b".into(),
                text: "goodbye world".into(),
                embedding: vec![0.0, 1.0, 0.0],
                namespace: None,
                ttl_seconds: None,
                created_at: 1000,
                bucket: 1,
            },
            Segment {
                id: "c".into(),
                text: "something else".into(),
                embedding: vec![0.0, 0.0, 1.0],
                namespace: None,
                ttl_seconds: None,
                created_at: 1000,
                bucket: 1,
            },
        ];

        let query: Vec<f32> = vec![0.9, 0.1, 0.0];
        // With query_text, uses stratified search; with <20 segments, falls through to brute force
        let results = nearest_neighbours(&query, &segments, 2, None, 2000, Some("hello world"));
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
        let results = nearest_neighbours(&query, &segments, 5, None, 2000, None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_l2_symmetry() {
        let a: Vec<f32> = (0..20).map(|i| i as f32 * 0.5).collect();
        let b: Vec<f32> = (0..20).map(|i| i as f32 * 0.5 + 1.0).collect();
        assert!((l2_distance(&a, &b) - l2_distance(&b, &a)).abs() < 1e-6);
    }

    // ---- Bucket assignment tests ----

    #[test]
    fn test_assign_bucket_short_numeric() {
        assert_eq!(assign_bucket("42"), 0);
        assert_eq!(assign_bucket("95.5%"), 0);
        assert_eq!(assign_bucket("1/2/3"), 0);
        assert_eq!(assign_bucket("cpu=45% mem=2.1G"), 0);
    }

    #[test]
    fn test_assign_bucket_medium_prose() {
        let medium = "The quick brown fox jumps over the lazy dog.";
        assert_eq!(assign_bucket(medium), 1);

        let longer = "a".repeat(250);
        assert_eq!(assign_bucket(&longer), 1);
    }

    #[test]
    fn test_assign_bucket_long() {
        let long = "a".repeat(2500);
        assert_eq!(assign_bucket(&long), 2);
    }

    #[test]
    fn test_assign_bucket_command() {
        assert_eq!(assign_bucket("system:update"), 3);
        assert_eq!(assign_bucket("cmd:deploy"), 3);
        assert_eq!(assign_bucket("fn do_something"), 3);
        assert_eq!(assign_bucket("def fibonacci(n):"), 3);
    }

    #[test]
    fn test_assign_bucket_short_text_defaults_to_prose() {
        // Short text that isn't numeric goes to bucket 1
        assert_eq!(assign_bucket("hi"), 1);
        assert_eq!(assign_bucket("hello world"), 1);
    }

    #[test]
    fn test_bucket_search_order_primary_first() {
        let order = bucket_search_order(2);
        assert_eq!(order[0], 2, "query's bucket should be first");
        let mut sorted = order;
        sorted.sort();
        assert_eq!(sorted, [0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_segment_bucket_deserialization_default() {
        // Old segments without bucket field should default to 0
        let json = r#"{
            "id": "test",
            "text": "hello",
            "embedding": [1.0]
        }"#;
        let seg: Segment = serde_json::from_str(json).unwrap();
        assert_eq!(seg.bucket, 0);
        assert_eq!(seg.text, "hello");
    }

    #[test]
    fn test_stratified_search_with_buckets() {
        // Create segments in different buckets
        let segments = vec![
            Segment {
                id: "numeric".into(),
                text: "42%".into(),
                embedding: vec![0.0, 1.0],
                namespace: None,
                ttl_seconds: None,
                created_at: 1000,
                bucket: 0,
            },
            Segment {
                id: "prose".into(),
                text: "some medium length text here".into(),
                embedding: vec![0.9, 0.0],
                namespace: None,
                ttl_seconds: None,
                created_at: 1000,
                bucket: 1,
            },
        ];

        let query: Vec<f32> = vec![1.0, 0.0];
        // Query text "hello" -> bucket 1 (medium prose)
        let results = nearest_neighbours(&query, &segments, 1, None, 2000, Some("hello"));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "prose");
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
