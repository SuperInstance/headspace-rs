//! headspace-rs — ARM-optimised vector embedding sidecar.
//!
//! Listens on `0.0.0.0:9090` and provides:
//!
//! - `POST /api/segment`  — store a text segment with embedding
//! - `POST /api/query`    — find top-k similar segments
//! - `POST /api/reset`    — clear all stored segments
//! - `GET  /api/status`   — number of segments stored
//!
//! State is written through to a flat JSON file (`store.json`) so it
//! survives restarts.

mod vector;

use axum::{
    Router,
    extract::{Path, State},
    http::StatusCode,
    routing::{delete, get, post},
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

use vector::{nearest_neighbours, Segment, SearchResult};

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

/// On-disk persistence path.
const STORE_PATH: &str = "store.json";

#[derive(Clone)]
struct AppState {
    segments: Arc<RwLock<Vec<Segment>>>,
}

impl AppState {
    /// Load segments from disk, returning an empty list on error.
    async fn from_disk() -> Self {
        let segments = match tokio::fs::read_to_string(STORE_PATH).await {
            Ok(s) => serde_json::from_str(&s).unwrap_or_default(),
            Err(_) => Vec::new(),
        };
        tracing::info!("loaded {} segments from disk", segments.len());
        Self {
            segments: Arc::new(RwLock::new(segments)),
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

/// POST /api/segment — store a new segment.
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
    let embedding = req.embedding.unwrap_or_else(|| {
        vec![0.0_f32; 384]
    });

    let id = Uuid::new_v4().to_string();
    let dims = embedding.len();
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    {
        let mut segs = state.segments.write().await;
        segs.push(Segment {
            id: id.clone(),
            text: text.clone(),
            embedding,
            namespace: req.namespace.clone(),
            ttl_seconds: req.ttl_seconds,
            created_at: now,
        });
    }
    state.persist().await;

    Ok(Json(SegmentResponse {
        id,
        text: req.text,
        dimensions: dims,
        namespace: req.namespace,
        ttl_seconds: req.ttl_seconds,
        created_at: now,
    }))
}

/// POST /api/query — find top-k similar segments.
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

    let segments = state.segments.read().await;

    if segments.is_empty() {
        return Ok(Json(QueryResponse {
            results: Vec::new(),
            query_text: req.text,
        }));
    }

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let results = nearest_neighbours(
        &req.embedding,
        &segments,
        req.k,
        req.namespace.as_deref(),
        now,
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
    let count = state.segments.read().await.len();
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
    let segments = state.segments.read().await;
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
    let deleted = {
        let mut segs = state.segments.write().await;
        let before = segs.len();
        segs.retain(|seg| seg.namespace.as_deref() != Some(&name));
        before - segs.len()
    };
    state.persist().await;
    Json(DeleteNamespaceResponse { deleted })
}

/// Purge TTL-expired segments from in-memory state and disk.
async fn purge_expired(state: &AppState) {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let count = {
        let mut segs = state.segments.write().await;
        let before = segs.len();
        segs.retain(|seg| {
            seg.ttl_seconds.map_or(true, |ttl| seg.created_at + ttl > now)
        });
        before - segs.len()
    };
    if count > 0 {
        tracing::info!("purged {count} expired segments");
        state.persist().await;
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    // Initialise structured logging.
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("headspace_rs=info")),
        )
        .init();

    let state = AppState::from_disk().await;

    // Purge expired segments on startup
    purge_expired(&state).await;

    let app = Router::new()
        .route("/api/segment", post(handle_segment))
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
    axum::serve(listener, app).await.unwrap();
}
