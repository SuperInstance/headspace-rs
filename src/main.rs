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
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
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
    /// Optional pre-computed embedding.  If absent, the server returns
    /// a placeholder embedding for the caller to fill (see design notes).
    #[serde(default)]
    embedding: Option<Vec<f32>>,
}

#[derive(Debug, Serialize)]
struct SegmentResponse {
    id: String,
    text: String,
    dimensions: usize,
}

#[derive(Debug, Deserialize)]
struct QueryRequest {
    text: String,
    /// Pre-computed query embedding (required).
    embedding: Vec<f32>,
    /// Number of results to return (default 5).
    #[serde(default = "default_k")]
    k: usize,
}

fn default_k() -> usize {
    5
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
    // The Python caller is expected to generate the actual embedding
    // and submit it via a separate call or with the segment.
    let embedding = req.embedding.unwrap_or_else(|| {
        // Default 384-dimensional placeholder (all-zeros)
        vec![0.0_f32; 384]
    });

    let id = Uuid::new_v4().to_string();
    let dims = embedding.len();

    {
        let mut segs = state.segments.write().await;
        segs.push(Segment {
            id: id.clone(),
            text,
            embedding,
        });
    }
    state.persist().await;

    Ok(Json(SegmentResponse {
        id,
        text: req.text,
        dimensions: dims,
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

    let results = nearest_neighbours(&req.embedding, &segments, req.k);

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
        api_version: "0.1.0",
    })
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

    let app = Router::new()
        .route("/api/segment", post(handle_segment))
        .route("/api/query", post(handle_query))
        .route("/api/reset", post(handle_reset))
        .route("/api/status", get(handle_status))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = "0.0.0.0:9090";
    tracing::info!("headspace-rs listening on {addr}");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
