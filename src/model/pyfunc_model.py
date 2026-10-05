"""Custom MLflow PyFunc that scores using scalar features + a Volume-backed JSON.

Request contract (per row / dataframe_records item):
  - var_01 .. var_15 : float
  - payload          : JSON object or JSON string (inline; used when present)
  - payload_uri      : str  (/Volumes/.../request_id.json) when payload is omitted
  - payload_hash     : str  (optional integrity check of the Volume file)
  - request_id       : str  (optional correlation id)

Inline `payload` is for tests/curl that send the JSON together with the scalars.
Production with inference-table limits still prefers Volume + payload_uri.
"""

from __future__ import annotations

from typing import Any

import mlflow
import pandas as pd
from mlflow.models import ModelSignature
from mlflow.types.schema import ColSpec, Schema

try:
    from .features import (  # type: ignore[import-not-found]
        REQUEST_COLUMNS,
        SCALAR_FEATURES,
        build_feature_frame,
        sha256_file,
    )
except ImportError:  # MLflow serving / notebook sys.path variants
    try:
        from model.features import (  # type: ignore[no-redef]
            REQUEST_COLUMNS,
            SCALAR_FEATURES,
            build_feature_frame,
            sha256_file,
        )
    except ImportError:
        from features import (  # type: ignore[no-redef]
            REQUEST_COLUMNS,
            SCALAR_FEATURES,
            build_feature_frame,
            sha256_file,
        )


class LargePayloadScorer(mlflow.pyfunc.PythonModel):
    """Wraps a sklearn classifier; loads JSON from UC Volume at predict-time."""

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        import joblib

        self.model = joblib.load(context.artifacts["sklearn_model"])

    def predict(self, context, model_input: pd.DataFrame, params: dict[str, Any] | None = None):
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)

        missing = [c for c in SCALAR_FEATURES if c not in model_input.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        has_inline = "payload" in model_input.columns
        has_uri = "payload_uri" in model_input.columns
        if not has_inline and not has_uri:
            raise ValueError("Provide inline `payload` and/or `payload_uri`")
        if not has_uri:
            model_input = model_input.copy()
            model_input["payload_uri"] = ""

        # Optional integrity check when the client sends payload_hash.
        if "payload_hash" in model_input.columns:
            for _, row in model_input.iterrows():
                expected = row.get("payload_hash")
                uri = row.get("payload_uri")
                if expected and uri and str(expected) not in ("", "nan", "None"):
                    actual = sha256_file(str(uri))
                    if actual != str(expected):
                        raise ValueError(
                            f"payload_hash mismatch for {uri}: expected {expected}, got {actual}"
                        )

        features = build_feature_frame(model_input)
        proba = self.model.predict_proba(features)[:, 1]
        label = (proba >= 0.5).astype(int)
        return pd.DataFrame(
            {
                "prediction": label,
                "probability": proba,
                "request_id": model_input["request_id"]
                if "request_id" in model_input.columns
                else [None] * len(model_input),
            }
        )


def model_signature() -> ModelSignature:
    # Keep all fields required for broad MLflow compatibility; clients may send "".
    input_schema = Schema(
        [ColSpec("double", c) for c in SCALAR_FEATURES]
        + [
            ColSpec("string", "payload_uri"),
            ColSpec("string", "payload_hash"),
            ColSpec("string", "request_id"),
            ColSpec("string", "payload"),
        ]
    )
    output_schema = Schema(
        [
            ColSpec("long", "prediction"),
            ColSpec("double", "probability"),
            ColSpec("string", "request_id"),
        ]
    )
    return ModelSignature(inputs=input_schema, outputs=output_schema)


def example_input_record() -> dict[str, Any]:
    rec = {c: 0.0 for c in SCALAR_FEATURES}
    rec.update(
        {
            "payload_uri": "/Volumes/ml_dev/mlops_large_payload/request_payloads/examples/demo.json",
            "payload_hash": "sha256:deadbeef",
            "request_id": "00000000-0000-0000-0000-000000000000",
            "payload": "",
        }
    )
    return rec


__all__ = [
    "LargePayloadScorer",
    "REQUEST_COLUMNS",
    "model_signature",
    "example_input_record",
]
