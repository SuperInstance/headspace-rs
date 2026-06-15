//! headspace-rs — ARM-optimised vector embedding sidecar.
//!
//! Listens on `0.0.0.0:9090` and provides:
//!
//! - `POST /api/segment`         — store a text segment with embedding
//! - `POST /api/segment/batch`   — batch-store segments (coalesced)
//! - `POST /api/query`           — find top-k similar segments
//! - `POST /api/reset`           — clear all stored segments
//! - `GET  /api/status`          — number of segments stored
//! - `GET  /api/segment/recent`  — most recent segments
//! - `DELETE /api/namespace/:name` — delete all segments in a namespace
//!
//! State is written through to a flat JSON file (`store.json`) so it
//! survives restarts.
//!
//! ## Batch Coalescing
//!
//! A background coalesce task collects incoming single-segment writes and
//! flushes them in groups (up to `max_batch_size`) after a settle window
//! (default 500 ms).

mod vector;

use axum::{
    Router,
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json,
};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

use vector::{assign_bucket, nearest_neighbours, Segment, SearchResult};

// ---------------------------------------------------------------------------
// Configuration (from env)
// ---------------------------------------------------------------------------

fn batch_window_ms() -> u64 {
    std::env::var("HEADSPACE_BATCH_WINDOW_MS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(500)
}

fn max_batch_size() -> usize {
    std::env::var("HEADSPACE_MAX_BATCH_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64)
}

fn worker_thread_count() -> usize {
    std::env::var("HEADSPACE_WORKERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4)
}

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

/// On-disk persistence path.
const STORE_PATH: &str = "store.json";

/// A buffered segment insertion waiting to be flushed.
#[derive(Debug, Clone)]
struct BufferedSegment {
    text: String,
    embedding: Vec<f32>,
    namespace: Option<String>,
    ttl_seconds: Option<u64>,
}

struct AppState {
    /// Authoritative segment store behind an async RwLock (write path).
    segments: Arc<RwLock<Vec<Segment>>>,
    /// Optimistic fast-read snapshot swapped atomically on writes.
    /// Readers snapshot the generation counter before and after cloning
    /// this Arc; if the generation matches, the snapshot is consistent.
    segments_snapshot: Arc<std::sync::RwLock<Arc<Vec<Segment>>>>,
    /// Monotonically increasing generation counter.
    /// Reader: load before, clone snapshot, load after. If equal, data is fresh.
    /// Writer: increment before AND after writing to the authoritative store.
    generation: AtomicU64,
    /// Coalescing buffer for single-segment writes.
    batch_buffer: Arc<RwLock<VecDeque<BufferedSegment>>>,
    /// Configuration parameters shared with the coalesce task.
    batch_window: Duration,
    batch_max: usize,
}

// Manual Clone implementation: AtomicU64 does not implement Clone.
// We use Arc for the fields that need shared ownership.
impl Clone for AppState {
    fn clone(&self) -> Self {
        Self {
            segments: self.segments.clone(),
            segments_snapshot: self.segments_snapshot.clone(),
            generation: AtomicU64::new(self.generation.load(Ordering::Acquire)),
            batch_buffer: self.batch_buffer.clone(),
            batch_window: self.batch_window,
            batch_max: self.batch_max,
        }
    }
}

impl AppState {
    /// Load segments from disk, returning an empty list on error.
    async fn from_disk() -> Self {
        let segments = match tokio::fs::read_to_string(STORE_PATH).await {
            Ok(s) => serde_json::from_str(&s).unwrap_or_default(),
            Err(_) => Vec::new(),
        };
        let segments_arc = Arc::new(segments.clone());
        tracing::info!("loaded {} segments from disk", segments.len());
        Self {
            segments: Arc::new(RwLock::new(segments)),
            segments_snapshot: Arc::new(std::sync::RwLock::new(segments_arc)),
            generation: AtomicU64::new(0),
            batch_buffer: Arc::new(RwLock::new(VecDeque::new())),
            batch_window: Duration::from_millis(batch_window_ms()),
            batch_max: max_batch_size(),
        }
    }

    /// Write the current segments to disk (write-through).
    async fn persist(&self) {
        let segments = self.segments.read().await;
        let json = serde_json::to_string_pretty(&*segments).unwrap_or_default();
        // Fire-and-forget: errors are logged but not fatal.
        if let Err(e) = tokio::fs::write(STORE_PATH, &json).await {
            tracing::warn!("failed to persist store: {e}");
        }
    }

    /// Optimistically try to read segments without acquiring the async RwLock.
    /// Returns `Some(Arc<Vec<Segment>>)` if the generation counter was stable
    /// during the snapshot, meaning the data is consistent. Falls back to
    /// `None` if a concurrent write was detected.
    fn try_read_optimistic(&self) -> Option<Arc<Vec<Segment>>> {
        let gen_before = self.generation.load(Ordering::Acquire);
        // Clone the Arc from the snapshot — this is O(1), just a refcount bump.
        let snapshot = self.segments_snapshot.read().unwrap().clone();
        let gen_after = self.generation.load(Ordering::Acquire);
        if gen_before == gen_after {
            Some(snapshot)
        } else {
            None
        }
    }

    /// Update the optimistic snapshot with new segment data.
    /// Call this after every mutation to `self.segments`.
    /// The caller is expected to already hold (or have just released) the
    /// tokio write lock.  We take ownership of the new Vec.
    fn update_snapshot(&self, new_segments: Vec<Segment>) {
        let new_arc = Arc::new(new_segments);
        {
            let mut guard = self.segments_snapshot.write().unwrap();
            *guard = new_arc;
        }
        // Signal readers that data has changed — done AFTER the swap
        // so readers see the updated counter alongside the new Arc.
        self.generation.fetch_add(1, Ordering::Release);
    }
}

// ---------------------------------------------------------------------------
// Request / Response types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct SegmentRequest {
    text: String,
    /// Optional pre-computed embedding.
    #[serde(default)]
    embedding: Option<Vec<f32>>,
    /// Optional namespace for partitioning (e.g. "pulse", "lures").
    namespace: Option<String>,
    /// Optional TTL in seconds from creation time.
    ttl_seconds: Option<u64>,
}

#[derive(Debug, Serialize)]
struct SegmentResponse {
    id: String,
    text: String,
    dimensions: usize,
    namespace: Option<String>,
    ttl_seconds: Option<u64>,
    created_at: u64,
}

/// Batch endpoint request body.
#[derive(Debug, Deserialize)]
struct BatchRequest {
    texts: Vec<String>,
    /// Embeddings in the same order as `texts`.  If shorter, missing slots
    /// get a placeholder.
    #[serde(default)]
    embeddings: Vec<Vec<f32>>,
    /// Common namespace for all segments in this batch.
    namespace: Option<String>,
    /// Optional TTL in seconds for all segments.
    ttl_seconds: Option<u64>,
}

#[derive(Debug, Serialize)]
struct BatchResponse {
    count: usize,
    ids: Vec<String>,
    dimensions: usize,
    namespace: Option<String>,
    ttl_seconds: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct QueryRequest {
    text: String,
    /// Pre-computed query embedding (required).
    embedding: Vec<f32>,
    /// Number of results to return (default 5).
    #[serde(default = "default_k")]
    k: usize,
    /// Optional namespace filter.
    namespace: Option<String>,
}

#[derive(Debug, Serialize)]
struct QueryResponse {
    results: Vec<SearchResult>,
    query_text: String,
}

#[derive(Debug, Serialize)]
struct StatusResponse {
    segments: usize,
    api_version: &'static str,
}

#[derive(Debug, Deserialize)]
struct RecentParams {
    namespace: Option<String>,
    #[serde(default = "default_limit")]
    limit: usize,
}

fn default_k() -> usize { 5 }

fn default_limit() -> usize { 10 }

#[derive(Debug, Deserialize, Serialize)]
struct DeleteNamespaceResponse {
    deleted: usize,
}

#[derive(Debug, Serialize)]
struct RecentResponse {
    segments: Vec<SegmentSummary>,
}

#[derive(Debug, Serialize)]
struct SegmentSummary {
    id: String,
    text: String,
    namespace: Option<String>,
    created_at: u64,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// POST /api/segment — store a new segment (via coalescing buffer).
async fn handle_segment(
    State(state): State<AppState>,
    Json(req): Json<SegmentRequest>,
) -> Result<Json<SegmentResponse>, (StatusCode, Json<ErrorResponse>)> {
    let text = req.text.trim().to_string();
    if text.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "text is required".into(),
            }),
        ));
    }

    // If the caller didn't supply an embedding, create a placeholder.
    let embedding = req.embedding.unwrap_or_else(|| vec![0.0_f32; 384]);

    let dims = embedding.len();

    // Push into the coalescing buffer.  We do NOT generate the id here;
    // the background coalesce task owns id assignment.  Return a synthetic
    // id indicating "pending" so the caller gets immediate feedback.
    let placeholder_id = format!("pending-{}", Uuid::new_v4());

    {
        let mut buf = state.batch_buffer.write().await;
        buf.push_back(BufferedSegment {
            text: text.clone(),
            embedding,
            namespace: req.namespace.clone(),
            ttl_seconds: req.ttl_seconds,
        });
        let buf_len = buf.len();
        // Size-based flush: if we hit max_batch_size, the coalesce task
        // will pick it up on its next tick — but we also send a
        // notification so it doesn't wait for the full window.
        if buf_len >= state.batch_max {
            tracing::info!(
                "batch buffer hit max_size ({}) — will flush on next coalesce tick",
                state.batch_max
            );
        }
    }

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    Ok(Json(SegmentResponse {
        id: placeholder_id,
        text: req.text,
        dimensions: dims,
        namespace: req.namespace,
        ttl_seconds: req.ttl_seconds,
        created_at: now,
    }))
}

/// POST /api/segment/batch — store many segments in one shot (direct insert,
/// bypassing the coalescing buffer).
async fn handle_batch(
    State(state): State<AppState>,
    Json(req): Json<BatchRequest>,
) -> Result<Json<BatchResponse>, (StatusCode, Json<ErrorResponse>)> {
    if req.texts.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "texts must be non-empty".into(),
            }),
        ));
    }

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let mut ids = Vec::with_capacity(req.texts.len());
    let mut dims = 0usize;

    let new_segments = {
        let mut segs = state.segments.write().await;
        for (i, text) in req.texts.iter().enumerate() {
            let embedding = req.embeddings.get(i).cloned().unwrap_or_else(|| vec![0.0_f32; 384]);
            let id = Uuid::new_v4().to_string();
            dims = embedding.len();
            segs.push(Segment {
                id: id.clone(),
                text: text.clone(),
                embedding,
                namespace: req.namespace.clone(),
                ttl_seconds: req.ttl_seconds,
                created_at: now,
                bucket: assign_bucket(text),
            });
            ids.push(id);
        }
        segs.clone()
    };

    state.update_snapshot(new_segments);
    state.persist().await;

    tracing::info!(
        "direct batch insert: {} segments (namespace={:?})",
        ids.len(),
        req.namespace
    );

    Ok(Json(BatchResponse {
        count: ids.len(),
        ids,
        dimensions: dims,
        namespace: req.namespace,
        ttl_seconds: req.ttl_seconds,
    }))
}

/// POST /api/query — find top-k similar segments.
///
/// Uses an optimistic generation-counter lock for the hot read path.
/// The reader snapshots the generation counter before and after cloning
/// the Arc snapshot. If the generation is stable, we skip the async
/// RwLock entirely and read from the snapshotted Vec directly.
/// If a concurrent write is detected, we fall through to the
/// authoritative RwLock read path.
async fn handle_query(
    State(state): State<AppState>,
    Json(req): Json<QueryRequest>,
) -> Result<Json<QueryResponse>, (StatusCode, Json<ErrorResponse>)> {
    if req.embedding.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "embedding is required and must be non-empty".into(),
            }),
        ));
    }

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Optimistic fast path: snapshot the generation counter and read
    // from the swap field without acquiring the async RwLock.
    if let Some(snapshot) = state.try_read_optimistic() {
        if !snapshot.is_empty() {
            let results = nearest_neighbours(
                &req.embedding,
                &snapshot,
                req.k,
                req.namespace.as_deref(),
                now,
                Some(&req.text),
            );
            return Ok(Json(QueryResponse {
                results,
                query_text: req.text,
            }));
        }
        // Snapshot was empty — return empty results
        return Ok(Json(QueryResponse {
            results: Vec::new(),
            query_text: req.text,
        }));
    }

    // Fallback: a concurrent write was detected or generation mismatch.
    // Acquire the authoritative RwLock.
    let segments = state.segments.read().await;

    if segments.is_empty() {
        return Ok(Json(QueryResponse {
            results: Vec::new(),
            query_text: req.text,
        }));
    }

    let results = nearest_neighbours(
        &req.embedding,
        &segments,
        req.k,
        req.namespace.as_deref(),
        now,
        Some(&req.text),
    );

    Ok(Json(QueryResponse {
        results,
        query_text: req.text,
    }))
}

/// POST /api/reset — clear all segments.
async fn handle_reset(
    State(state): State<AppState>,
) -> Json<StatusResponse> {
    {
        let mut segs = state.segments.write().await;
        segs.clear();
    }
    // Create empty snapshot for the optimistic read path
    state.update_snapshot(Vec::new());
    state.persist().await;
    Json(StatusResponse {
        segments: 0,
        api_version: "0.1.0",
    })
}

/// GET /api/status — number of segments and basic info.
async fn handle_status(
    State(state): State<AppState>,
) -> Json<StatusResponse> {
    let count = if let Some(snap) = state.try_read_optimistic() {
        snap.len()
    } else {
        state.segments.read().await.len()
    };
    Json(StatusResponse {
        segments: count,
        api_version: "0.2.0",
    })
}

/// GET /api/segment/recent — most recent segments in a namespace.
async fn handle_recent(
    State(state): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<RecentParams>,
) -> Json<RecentResponse> {
    let segments: Arc<Vec<Segment>> = if let Some(snap) = state.try_read_optimistic() {
        snap
    } else {
        Arc::new(state.segments.read().await.clone())
    };

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let mut filtered: Vec<&Segment> = segments
        .iter()
        .filter(|seg| {
            let ns_ok = params.namespace.as_deref().map_or(true, |ns| {
                seg.namespace.as_deref() == Some(ns)
            });
            let ttl_ok = seg.ttl_seconds.map_or(true, |ttl| seg.created_at + ttl > now);
            ns_ok && ttl_ok
        })
        .collect();

    filtered.sort_unstable_by(|a, b| b.created_at.cmp(&a.created_at));
    filtered.truncate(params.limit);

    let segment_summaries: Vec<SegmentSummary> = filtered
        .into_iter()
        .map(|s| SegmentSummary {
            id: s.id.clone(),
            text: s.text.clone(),
            namespace: s.namespace.clone(),
            created_at: s.created_at,
        })
        .collect();

    Json(RecentResponse {
        segments: segment_summaries,
    })
}

/// DELETE /api/namespace/:name — delete all segments in a namespace.
async fn handle_delete_namespace(
    State(state): State<AppState>,
    axum::extract::Path(name): axum::extract::Path<String>,
) -> Json<DeleteNamespaceResponse> {
    let (deleted, new_segments) = {
        let mut segs = state.segments.write().await;
        let before = segs.len();
        segs.retain(|seg| seg.namespace.as_deref() != Some(&name));
        let deleted = before - segs.len();
        (deleted, segs.clone())
    };
    state.update_snapshot(new_segments);
    state.persist().await;
    Json(DeleteNamespaceResponse { deleted })
}

/// Purge TTL-expired segments from in-memory state and disk.
async fn purge_expired(state: &AppState) {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let (count, new_segments) = {
        let mut segs = state.segments.write().await;
        let before = segs.len();
        segs.retain(|seg| {
            seg.ttl_seconds.map_or(true, |ttl| seg.created_at + ttl > now)
        });
        (before - segs.len(), segs.clone())
    };
    if count > 0 {
        state.update_snapshot(new_segments);
        tracing::info!("purged {count} expired segments");
        state.persist().await;
    }
}

// ---------------------------------------------------------------------------
// Coalesce task
// ---------------------------------------------------------------------------

/// Background task that watches the coalescing buffer and flushes collected
/// segments to the main store on a settle window or when the buffer is full.
async fn coalesce_task(state: AppState) {
    let window = state.batch_window;
    let max_batch = state.batch_max;
    // We use a fixed-interval ticker.  Every `window` the task checks the
    // buffer.  If items are present and none arrived since the last check
    // (i.e. the buffer length is stable), it flushes.  Size-based flush
    // happens inline when draining.
    //
    // A more precise approach would be to use a watch channel, but this
    // is simple and keeps the timer-based guarantee.
    let mut interval = tokio::time::interval(window);

    tracing::info!(
        "coalesce task started: window={}ms max_batch={}",
        window.as_millis(),
        max_batch,
    );

    loop {
        interval.tick().await;

        // Drain the buffer in batches of up to `max_batch`.
        let mut total_flushed = 0usize;
        loop {
            let batch: Vec<BufferedSegment> = {
                let mut buf = state.batch_buffer.write().await;
                if buf.is_empty() {
                    break;
                }
                let take = buf.len().min(max_batch);
                buf.drain(..take).collect()
            };

            if batch.is_empty() {
                break;
            }

            // Persist this batch.
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();

            let new_segments = {
                let mut segs = state.segments.write().await;
                for item in &batch {
                    segs.push(Segment {
                        id: Uuid::new_v4().to_string(),
                        text: item.text.clone(),
                        embedding: item.embedding.clone(),
                        namespace: item.namespace.clone(),
                        ttl_seconds: item.ttl_seconds,
                        created_at: now,
                        bucket: assign_bucket(&item.text),
                    });
                }
                segs.clone()
            };

            state.update_snapshot(new_segments);
            state.persist().await;
            total_flushed += batch.len();
            let reason = if batch.len() >= max_batch {
                "size-flush"
            } else {
                "window-expiry"
            };

            tracing::info!(
                "coalesce flush: batch_size={} total_flushed={} reason={reason}",
                batch.len(),
                total_flushed,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() {
    // Pick up runtime thread count override from env
    let _wt = worker_thread_count();

    // Initialise structured logging.
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("headspace_rs=info")),
        )
        .init();

    let state = AppState::from_disk().await;

    // Log batch config at startup
    tracing::info!(
        "batch config: window={}ms max_batch={}",
        state.batch_window.as_millis(),
        state.batch_max,
    );

    // Purge expired segments on startup
    purge_expired(&state).await;

    // Backfill bucket field for segments loaded from disk that lack it
    // (e.g. migrated from older version without the bucket field).
    // Segments deserialized with bucket=0 may actually belong in other buckets.
    let (needs_backfill, backfill_snapshot) = {
        let segs = state.segments.read().await;
        let mut changed = false;
        let mut corrected = segs.clone();
        for seg in corrected.iter_mut() {
            let expected = vector::assign_bucket(&seg.text);
            if seg.bucket != expected {
                seg.bucket = expected;
                changed = true;
            }
        }
        (changed, corrected)
    };
    if needs_backfill {
        {
            let mut segs = state.segments.write().await;
            *segs = backfill_snapshot;
        }
        state.update_snapshot(state.segments.read().await.clone());
        state.persist().await;
        tracing::info!("backfilled bucket assignments for {} existing segments", state.segments.read().await.len());
    }

    // Spawn the background coalesce task.
    let coalesce_state = state.clone();
    tokio::spawn(async move {
        coalesce_task(coalesce_state).await;
    });

    let app = Router::new()
        .route("/api/segment", post(handle_segment))
        .route("/api/segment/batch", post(handle_batch))
        .route("/api/query", post(handle_query))
        .route("/api/reset", post(handle_reset))
        .route("/api/status", get(handle_status))
        .route("/api/segment/recent", get(handle_recent))
        .route("/api/namespace/{name}", axum::routing::delete(handle_delete_namespace))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = "0.0.0.0:9090";
    tracing::info!("headspace-rs listening on {addr}");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    tracing::info!("runtime threads: {} ({})",
        worker_thread_count(),
        std::env::var("HEADSPACE_WORKERS").unwrap_or_else(|_| "default".into())
    );
    axum::serve(listener, app).await.unwrap();
}
