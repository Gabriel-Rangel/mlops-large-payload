"""In-memory request status store (per-process; lost on app restart)."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any

_lock = threading.Lock()
_jobs: dict[str, dict[str, Any]] = {}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_job(request_id: str, extras: dict[str, Any] | None = None) -> dict[str, Any]:
    job: dict[str, Any] = {
        "request_id": request_id,
        "status": "queued",
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "payload_uri": None,
        "payload_hash": None,
        "payload_bytes": None,
        "prediction": None,
        "probability": None,
        "latency_ms": None,
        "accept_latency_ms": None,
        "total_latency_ms": None,
        "score_latency_ms": None,
        "error_message": None,
        "model_version": None,
    }
    if extras:
        job.update(extras)
    with _lock:
        _jobs[request_id] = job
    return dict(job)


def update_job(request_id: str, **fields: Any) -> dict[str, Any] | None:
    with _lock:
        job = _jobs.get(request_id)
        if job is None:
            return None
        job.update(fields)
        job["updated_at"] = now_iso()
        return dict(job)


def get_job(request_id: str) -> dict[str, Any] | None:
    with _lock:
        job = _jobs.get(request_id)
        return dict(job) if job else None
