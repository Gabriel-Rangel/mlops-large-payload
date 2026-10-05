"""Client-side helpers: persist large JSON to a UC Volume, then call serving.

Typical production sequence:
  1. Caller generates request_id.
  2. Write the ~16 MB JSON to /Volumes/<cat>/<schema>/request_payloads/...
  3. POST to Model Serving with the 15 scalars + payload_uri (+ hash).
  4. Append a row to the observability Delta table (metadata only).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.model.features import SCALAR_FEATURES, sha256_bytes


def build_payload_path(
    catalog: str,
    schema: str,
    volume: str = "request_payloads",
    request_id: str | None = None,
    dt: datetime | None = None,
) -> tuple[str, str]:
    """Return (request_id, absolute Volume path) partitioned by date."""
    request_id = request_id or str(uuid.uuid4())
    dt = dt or datetime.now(timezone.utc)
    day = dt.strftime("%Y-%m-%d")
    uri = f"/Volumes/{catalog}/{schema}/{volume}/dt={day}/{request_id}.json"
    return request_id, uri


def write_payload_to_volume(payload: dict[str, Any], payload_uri: str) -> dict[str, Any]:
    """Serialize JSON to a UC Volume path (must be mounted / accessible)."""
    path = Path(payload_uri)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    path.write_bytes(raw)
    return {
        "payload_uri": payload_uri,
        "payload_hash": sha256_bytes(raw),
        "payload_bytes": len(raw),
    }


def build_serving_record(
    scalars: dict[str, float],
    payload_uri: str,
    payload_hash: str,
    request_id: str,
) -> dict[str, Any]:
    """Build one dataframe_records item for Model Serving (no raw JSON)."""
    missing = [c for c in SCALAR_FEATURES if c not in scalars]
    if missing:
        raise ValueError(f"Missing scalar features: {missing}")
    record = {c: float(scalars[c]) for c in SCALAR_FEATURES}
    record.update(
        {
            "payload_uri": payload_uri,
            "payload_hash": payload_hash,
            "request_id": request_id,
            "payload": "",
        }
    )
    return record


def observability_row(
    *,
    request_id: str,
    endpoint_name: str,
    model_version: str | None,
    payload_uri: str,
    payload_hash: str,
    payload_bytes: int,
    status: str,
    latency_ms: float | None,
    prediction: int | None = None,
    probability: float | None = None,
    error_message: str | None = None,
    accept_latency_ms: float | None = None,
    total_latency_ms: float | None = None,
) -> dict[str, Any]:
    """Metadata-only row for the custom observability Delta table."""
    total = total_latency_ms if total_latency_ms is not None else latency_ms
    return {
        "request_id": request_id,
        "event_ts": datetime.now(timezone.utc).isoformat(),
        "endpoint_name": endpoint_name,
        "model_version": model_version,
        "status": status,
        "latency_ms": total,
        "accept_latency_ms": accept_latency_ms,
        "total_latency_ms": total,
        "payload_uri": payload_uri,
        "payload_hash": payload_hash,
        "payload_bytes": payload_bytes,
        "prediction": prediction,
        "probability": probability,
        "error_message": error_message,
    }
