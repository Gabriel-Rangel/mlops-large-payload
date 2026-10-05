"""Feature helpers for the large-payload scoring demo.

The real use-case has ~15 scalar variables plus one large JSON (~16 MB).
In production the JSON must NOT travel inside the Model Serving request.
It lives in a UC Volume; the model (or a pre-processor) loads it by URI.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Scalar feature columns expected by the model (demo names — rename to match the customer).
SCALAR_FEATURES = [
    "var_01",
    "var_02",
    "var_03",
    "var_04",
    "var_05",
    "var_06",
    "var_07",
    "var_08",
    "var_09",
    "var_10",
    "var_11",
    "var_12",
    "var_13",
    "var_14",
    "var_15",
]

# Request columns sent to the endpoint (scalars + reference, never the raw JSON blob).
REQUEST_COLUMNS = SCALAR_FEATURES + ["payload_uri", "payload_hash", "request_id", "payload"]

# Keys extracted from the large JSON and folded into the feature vector.
JSON_DERIVED_FEATURES = [
    "json_n_items",
    "json_total_amount",
    "json_avg_amount",
    "json_max_amount",
    "json_unique_categories",
]


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


_ws_client = None


def _workspace_client():
    """Lazy WorkspaceClient — Jobs mount /Volumes; Model Serving does not."""
    global _ws_client
    if _ws_client is None:
        from databricks.sdk import WorkspaceClient

        _ws_client = WorkspaceClient()
    return _ws_client


def read_payload_bytes(payload_uri: str) -> bytes:
    """Read payload bytes from POSIX /Volumes (Jobs) or UC Files API (Serving).

    Custom Model Serving containers do not FUSE-mount UC Volumes, so `open(/Volumes/...)`
    fails even when the serving identity has READ VOLUME. The Files API uses the
    injected serving credentials and works with that grant.
    """
    if not payload_uri:
        raise ValueError("payload_uri is required")

    path = Path(payload_uri)
    if path.exists():
        return path.read_bytes()

    if not str(payload_uri).startswith("/Volumes/"):
        raise FileNotFoundError(f"Payload not found at {payload_uri}")

    try:
        response = _workspace_client().files.download(payload_uri)
        stream = response.contents
        if hasattr(stream, "read"):
            try:
                return stream.read()
            finally:
                close = getattr(stream, "close", None)
                if callable(close):
                    close()
        return bytes(stream)
    except Exception as exc:  # noqa: BLE001 — surface as FileNotFound for the scorer
        raise FileNotFoundError(
            f"Could not read {payload_uri} via UC Files API. "
            "Grant READ VOLUME (plus USE CATALOG / USE SCHEMA) to the Model Serving "
            f"identity. Underlying error: {exc}"
        ) from exc


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(read_payload_bytes(str(path)))


def load_json_payload(payload_uri: str) -> dict[str, Any]:
    """Load a JSON document from a UC Volume path or local filesystem path.

    Expected form: /Volumes/<catalog>/<schema>/<volume>/.../file.json
    """
    return json.loads(read_payload_bytes(payload_uri).decode("utf-8"))


def derive_features_from_json(payload: dict[str, Any]) -> dict[str, float]:
    """Reduce the large JSON to a small, model-friendly feature set.

    Replace this with the customer's real feature engineering. The point of
    the reference pattern is: heavy parsing happens against Volume storage,
    not against a 16 MB HTTP body.
    """
    items = payload.get("items") or []
    amounts = [float(i.get("amount", 0.0)) for i in items if isinstance(i, dict)]
    categories = {
        str(i.get("category"))
        for i in items
        if isinstance(i, dict) and i.get("category") is not None
    }

    n = len(amounts)
    total = float(sum(amounts)) if amounts else 0.0
    avg = total / n if n else 0.0
    mx = float(max(amounts)) if amounts else 0.0

    return {
        "json_n_items": float(n),
        "json_total_amount": total,
        "json_avg_amount": avg,
        "json_max_amount": mx,
        "json_unique_categories": float(len(categories)),
    }


def _inline_payload(rec: Any) -> dict[str, Any] | None:
    """JSON enviado no próprio request (`payload`), se houver."""
    if "payload" not in getattr(rec, "index", []):
        return None
    raw = rec.get("payload")
    if raw is None:
        return None
    if isinstance(raw, float) and np.isnan(raw):
        return None
    if isinstance(raw, dict):
        return raw
    text = str(raw)
    if text in ("", "nan", "None"):
        return None
    return json.loads(text)


def build_feature_frame(records: pd.DataFrame) -> pd.DataFrame:
    """Build the numeric matrix used by the sklearn model.

    For each row: scalar vars + features from inline `payload` (preferred)
    or from the JSON at `payload_uri`.
    """
    rows: list[dict[str, float]] = []
    for _, rec in records.iterrows():
        scalars = {c: float(rec.get(c, 0.0) or 0.0) for c in SCALAR_FEATURES}
        inline = _inline_payload(rec)
        uri = rec.get("payload_uri")
        uri_ok = not (
            uri is None or (isinstance(uri, float) and np.isnan(uri)) or str(uri) == ""
        )
        if inline is not None:
            derived = derive_features_from_json(inline)
        elif uri_ok:
            derived = derive_features_from_json(load_json_payload(str(uri)))
        else:
            derived = {k: 0.0 for k in JSON_DERIVED_FEATURES}
        rows.append({**scalars, **derived})
    return pd.DataFrame(rows, columns=SCALAR_FEATURES + JSON_DERIVED_FEATURES)


def make_synthetic_training_frame(n: int = 400, seed: int = 7) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic labeled data for the demo train job (no customer data required)."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(n, len(SCALAR_FEATURES) + len(JSON_DERIVED_FEATURES))),
        columns=SCALAR_FEATURES + JSON_DERIVED_FEATURES,
    )
    # Simple linear decision boundary + noise → binary label
    score = (
        0.4 * X["var_01"]
        + 0.3 * X["var_02"]
        + 0.2 * X["json_total_amount"]
        + 0.15 * X["json_n_items"]
        + rng.normal(scale=0.5, size=n)
    )
    y = pd.Series((score > 0).astype(int), name="label")
    return X, y
