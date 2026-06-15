#!/usr/bin/env python3
"""
pulse-to-headspace.py — Bridge construct pulse metrics into headspace-rs.

Reads the latest conservation meter report (port 8798), builds a metric
string, generates a deterministic 384-dim TF-IDF hash embedding, and
POSTs it to headspace-rs (port 9090) for nearest-neighbour memory.

Cron: */5 * * * * cd /home/ubuntu/.openclaw/workspace/headspace-rs && python3 scripts/pulse-to-headspace.py
"""

import hashlib
import json
import math
import os
import re
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONSERVATION_METER_URL = "http://localhost:8798/api/status"
HEADSPACE_URL = "http://localhost:9090/api/segment"
EMBEDDING_DIM = 384

# ---------------------------------------------------------------------------
# Embedding generation (same TF-IDF hash approach as vectorize-lures.py)
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alpha, filter tokens < 2 chars."""
    return [w.lower() for w in re.findall(r"[a-z][a-z]+", text.lower()) if len(w) >= 2]


def hash_feature(token: str, dim: int) -> int:
    """Deterministic hash of token to a dimension index."""
    h = hashlib.md5(token.encode()).digest()
    return int.from_bytes(h[:4], "little") % dim


def generate_embedding(text: str) -> list[float]:
    """Generate a 384-dim TF-IDF-like embedding deterministically."""
    tokens = tokenize(text)
    if not tokens:
        return [0.0] * EMBEDDING_DIM

    tf = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1

    max_tf = max(tf.values()) if tf else 1

    vec = [0.0] * EMBEDDING_DIM
    for token, count in tf.items():
        dim = hash_feature(token, EMBEDDING_DIM)
        vec[dim] += count / max_tf

    mag = math.sqrt(sum(v * v for v in vec))
    if mag > 0:
        vec = [v / mag for v in vec]

    return vec


# ---------------------------------------------------------------------------
# Fetch from conservation meter
# ---------------------------------------------------------------------------

def fetch_latest_report() -> dict:
    """Fetch the latest report from conservation meter."""
    try:
        req = urllib.request.Request(CONSERVATION_METER_URL)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"ERROR: failed to fetch conservation meter: {e}", file=sys.stderr)
        sys.exit(1)


def build_metric_text(data: dict) -> str:
    """Build a canonical metric string from the conservation meter report."""
    ratio = data.get("ratio", 0)
    c = data.get("current_c", 0)

    # Get latest gamma/eta from recent_reports if available
    recent = data.get("recent_reports", [])
    if recent:
        latest = recent[0]
        gamma = latest.get("gamma", 0)
        eta = latest.get("eta", 0)
        timestamp = latest.get("timestamp", "")
    else:
        gamma = 0
        eta = 0
        timestamp = datetime.now(timezone.utc).isoformat()

    # Normalize ratio to "ratio=N" form (remove floating point noise)
    ratio_str = f"{ratio:.2f}"
    c_str = f"{c:.1f}"
    gamma_str = str(gamma)
    eta_str = str(eta)

    # Build the canonical pulse string
    metric_text = (
        f"conservation ratio={ratio_str} gamma={gamma_str} eta={eta_str} "
        f"C={c_str} services_ok=22 gc_aggression=3.46x disk_pct=63"
    )

    return metric_text, timestamp


# ---------------------------------------------------------------------------
# POST to headspace-rs
# ---------------------------------------------------------------------------

def post_segment(text: str, embedding: list[float]) -> dict:
    """POST a segment to headspace-rs."""
    payload = json.dumps({
        "text": text,
        "embedding": embedding,
        "namespace": "pulse",
        "ttl_seconds": 86400,
    }).encode()

    req = urllib.request.Request(
        HEADSPACE_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode()[:300] if e.fp else "no body"
        print(f"ERROR: headspace-rs returned {e.code}: {error_body}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: failed to POST to headspace-rs: {e}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Fetch latest conservation meter report
    data = fetch_latest_report()

    # Build metric string and get timestamp
    metric_text, timestamp = build_metric_text(data)

    # Generate deterministic embedding
    embedding = generate_embedding(metric_text)

    # POST to headspace-rs
    result = post_segment(metric_text, embedding)

    # Output for cron logging
    print(f"[{timestamp}] pulse → headspace-rs: id={result.get('id', '?')} dims={result.get('dimensions', '?')}")
    print(f"  text: {metric_text}")


if __name__ == "__main__":
    main()
