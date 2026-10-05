"""Persist payload bytes to a UC Volume and append observability metadata."""

from __future__ import annotations

import io
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from databricks.sdk import WorkspaceClient

logger = logging.getLogger("large-payload-api.storage")


def workspace_client() -> WorkspaceClient:
    host = os.environ.get("DATABRICKS_HOST", "")
    if host and not host.startswith("http"):
        os.environ["DATABRICKS_HOST"] = f"https://{host}"
    return WorkspaceClient()


def build_payload_path(
    catalog: str,
    schema: str,
    volume: str,
    request_id: str,
    dt: datetime | None = None,
) -> str:
    day = (dt or datetime.now(timezone.utc)).strftime("%Y-%m-%d")
    return f"/Volumes/{catalog}/{schema}/{volume}/dt={day}/{request_id}.json"


def write_payload_bytes(payload_uri: str, raw: bytes) -> None:
    """Write JSON bytes to a UC Volume via POSIX if mounted, else Files API."""
    path = Path(payload_uri)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        logger.info("Wrote payload via POSIX path (%s bytes)", len(raw))
        return
    except OSError as posix_exc:
        logger.info("POSIX /Volumes not writable (%s); using Files API", posix_exc)

    w = workspace_client()
    parent = str(path.parent)
    try:
        w.files.create_directory(parent)
    except Exception as exc:  # noqa: BLE001
        logger.debug("create_directory %s: %s", parent, exc)

    w.files.upload(payload_uri, io.BytesIO(raw), overwrite=True)
    logger.info("Wrote payload via Files API (%s bytes)", len(raw))


def _sql_str(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value != value:  # NaN
            return "NULL"
        return str(value)
    text = str(value).replace("'", "''")
    return f"'{text}'"


def insert_observability_row(row: dict[str, Any]) -> None:
    warehouse_id = os.environ.get("DATABRICKS_WAREHOUSE_ID")
    table = os.environ.get(
        "OBSERVABILITY_TABLE",
        "gabriel_caballeria_demo_catalog.mlops_large_payload.inference_observability",
    )
    if not warehouse_id:
        raise RuntimeError("DATABRICKS_WAREHOUSE_ID is not set")

    columns = [
        "request_id",
        "event_ts",
        "endpoint_name",
        "model_version",
        "model_alias",
        "status",
        "latency_ms",
        "accept_latency_ms",
        "total_latency_ms",
        "payload_uri",
        "payload_hash",
        "payload_bytes",
        "prediction",
        "probability",
        "error_message",
        "env",
        "ingested_at",
        "scalars_json",
    ]
    values_sql = ", ".join(
        [
            _sql_str(row.get("request_id")),
            "current_timestamp()",
            _sql_str(row.get("endpoint_name")),
            _sql_str(row.get("model_version")),
            _sql_str(row.get("model_alias")),
            _sql_str(row.get("status")),
            _sql_str(row.get("latency_ms")),
            _sql_str(row.get("accept_latency_ms")),
            _sql_str(row.get("total_latency_ms")),
            _sql_str(row.get("payload_uri")),
            _sql_str(row.get("payload_hash")),
            _sql_str(row.get("payload_bytes")),
            _sql_str(row.get("prediction")),
            _sql_str(row.get("probability")),
            _sql_str(row.get("error_message")),
            _sql_str(row.get("env")),
            "current_timestamp()",
            _sql_str(row.get("scalars_json")),
        ]
    )
    statement = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({values_sql})"
    w = workspace_client()
    resp = w.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=statement,
        wait_timeout="50s",
    )
    state = getattr(getattr(resp, "status", None), "state", None)
    state_name = getattr(state, "value", state)
    logger.info(
        "Observability insert request_id=%s statement=%s state=%s",
        row.get("request_id"),
        getattr(resp, "statement_id", None),
        state_name,
    )
    if str(state_name).upper() in {"FAILED", "CANCELED", "CLOSED"}:
        error = getattr(getattr(resp, "status", None), "error", None)
        raise RuntimeError(f"Observability INSERT failed: {error or state_name}")
