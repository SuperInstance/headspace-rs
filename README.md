# headspace-rs

ARM-optimised vector-embedding sidecar for [headspace](https://github.com/SuperInstance/headspace).

Replaces the Python embedding/segmentation path with a Rust daemon that uses **NEON SIMD intrinsics** (`vld1q_f32`, `vfmaq_f32`, `vpaddq_f32`) for maximum throughput on AArch64.

## Quick start

```bash
cargo run --release
```

Server starts on `0.0.0.0:9090`.

## API

### `POST /api/segment` — store a text segment

```json
{"text": "...", "embedding": [0.1, 0.2, ...]}
```

Returns `{"id": "<uuid>", "text": "...", "dimensions": 384}`.

### `POST /api/query` — find top-k similar

```json
{"text": "...", "embedding": [0.1, 0.2, ...], "k": 5}
```

Returns `{"results": [{"id": "..", "text": "..", "score": 0.99}, ...]}`.

### `POST /api/reset` — clear store

Returns `{"segments": 0}`.

### `GET /api/status` — segment count

Returns `{"segments": N, "api_version": "0.1.0"}`.

## Architecture

```
┌──────────────┐      POST /segment      ┌──────────────────┐
│  Python       │ ──────────────────────▶  │  headspace-rs    │
│  headspace    │                          │  (axum server)   │
│  (embedding)  │ ◀────────────────────── │  0.0.0.0:9090    │
└──────────────┘      POST /query          └───────┬──────────┘
                                                    │
                                          ┌─────────▼────────┐
                                          │  store.json       │
                                          │  (write-through)  │
                                          └───────────────────┘
```

The Python side calls an embedding model (e.g. `sentence-transformers`), then ships the vectors to headspace-rs for storage and retrieval. headspace-rs handles all the heavy vector arithmetic with NEON SIMD.

## ARM optimisation

The dot product uses:

- **`vld1q_f32`** — 128-bit NEON load (4× f32)
- **`vfmaq_f32`** — fused multiply-add (FMA), the key throughput path
- **`vpaddq_f32`** — horizontal pair-wise addition for reduction
- **FMA µop fusion** — Neoverse N1 can dispatch 2 FMA µops / cycle → ~8 FLOP/cycle

Build with the Neoverse N1 target for maximum efficiency:

```bash
RUSTFLAGS="-C target-cpu=neoverse-n1" cargo build --release
```

This is already configured in `.cargo/config.toml`.

## Stubs / future work

- No query-side segmentation model (Python provides embeddings)
- No ANN index (brute-force linear scan — fine for prototype scale)
- No auth / TLS
- No metrics endpoint
